import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rc.models.word_model import WordModel
from rc.utils.eval_utils import compute_eval_metric
from rc.models.module import multi_nll_loss
from rc.utils import constants as Constants
from collections import Counter
from rc.models.rnet import RNet


class Model(object):
    """High level models that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        self.config = config
        self.char_dict = self.init_char_dict()
        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            assert train_set is not None
            if not config['use_elmo']:
                print('Train vocab: {}'.format(len(train_set.vocab)))
                vocab = Counter()
                for w in train_set.vocab:
                    if train_set.vocab[w] >= config['min_freq']:
                        vocab[w] = train_set.vocab[w]
                print('Pruned train vocab: {}'.format(len(vocab)))
                # Building network.
                word_model = WordModel(additional_vocab=vocab,
                                       embed_size=self.config['embed_size'],
                                       filename=self.config['embed_file'],
                                       embed_type=self.config['embed_type'])
                self.config['embed_size'] = word_model.embed_size
                self.word_dict = word_model.get_vocab()
            else:
                self.word_dict = Counter()
                word_model = None
            self.network = RNet(self.config, word_model)

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()
        print('#Parameters = {}\n'.format(num_params))

        self._init_optimizer()

    def init_char_dict(self):
        # char vocabulary
        char_voc = {Constants._UNK_CHAR: 1}
        for char in Constants._ALPHABETS:
            char_voc[char] = len(char_voc) + 1
        return char_voc

    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['embed_size', 'hidden_size', 'max_word_length', 'sent_rnn_layers', 'use_elmo',
                      'sum_loss', 'fix_embeddings', 'dropout_rnn', 'dropout_emb']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved models %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.word_dict = saved_params['word_dict']
        state_dict = saved_params['state_dict']
        for k in _ARGUMENTS:
            if saved_params['config'][k] != self.config[k]:
                print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
                self.config[k] = saved_params['config'][k]

        if not self.config['use_elmo']:
            word_model = WordModel(additional_vocab=self.word_dict,
                                   embed_size=self.config['embed_size'])
        else:
            word_model = None
        self.network = RNet(self.config, word_model)

        # Merge the arguments
        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(parameters, rho=0.95,
                                            weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])

    def predict(self, ex, update=True, out_predictions=False):
        # Train/Eval mode
        self.network.train(update)
        # Run forward
        res = self.network(ex)
        score_s, score_e = res['score_s'], res['score_e']

        output = {
            'f1': 0.0,
            'em': 0.0,
            'loss': 0.0
        }
        # Loss cannot be computed for test-time as we may not have targets
        if update:
            # Compute loss and accuracies
            loss = self.compute_span_loss(score_s, score_e, res['targets'])
            output['loss'] = loss.item()

            # Clear gradients and run backward
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if not self.config['use_multi_gpu']:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['grad_clipping'])
            else:
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), self.config['grad_clipping'])

            # Update parameters
            self.optimizer.step()

        if (not update) or self.config['predict_train']:
            predictions, spans = self.extract_predictions(ex, score_s, score_e)
            output['f1'], output['em'] = self.evaluate_predictions(predictions, ex['answers'])
            if out_predictions:
                output['predictions'] = predictions
                output['spans'] = spans
        return output

    def compute_span_loss(self, score_s, score_e, targets):
        assert targets.size(0) == score_s.size(0) == score_e.size(0)
        if self.config['sum_loss']:
            loss = multi_nll_loss(score_s, targets[:, :, 0]) + multi_nll_loss(score_e, targets[:, :, 1])
        else:
            loss = F.nll_loss(score_s, targets[:, 0]) + F.nll_loss(score_e, targets[:, 1])
        return loss

    def extract_predictions(self, ex, score_s, score_e):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        score_s = score_s.exp().squeeze()
        score_e = score_e.exp().squeeze()

        predictions = []
        spans = []
        for i, (_s, _e) in enumerate(zip(score_s, score_e)):
            prediction, span = self._scores_to_text(ex['evidence_text'][i], _s, _e)
            predictions.append(prediction)
            spans.append(span)
        return predictions, spans

    def _scores_to_text(self, text, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))

    def _scores_to_raw_text(self, raw_text, offsets, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return raw_text[offsets[s_idx][0]: offsets[e_idx][1]], (offsets[s_idx][0], offsets[e_idx][1])

    def evaluate_predictions(self, predictions, answers):
        f1_score = compute_eval_metric('f1', predictions, answers)
        em_score = compute_eval_metric('em', predictions, answers)
        return f1_score, em_score

    def save(self, dirname):
        if not self.config['use_multi_gpu']:
            params = {
                'state_dict': {
                    'network': self.network.state_dict()
                },
                'word_dict': self.word_dict,
                'config': self.config,
                'dir': dirname
            }
        else:
            params = {
                'state_dict': {
                    'network': self.network.module.state_dict()     # multiGPU setting
                },
                'word_dict': self.word_dict,
                'config': self.config,
                'dir': dirname
            }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')
