# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import io
import torch
import numpy as np

from collections import Counter, defaultdict
from torch.utils.data import Dataset
from . import constants as Constants
from .timer import Timer
from allennlp.modules.elmo import batch_to_ids

################################################################################
# Dataset Prep #
################################################################################


def prepare_datasets(config):
    train_set = None if config['trainset'] is None else CoQADataset(config['trainset'], config, training=True)
    dev_set = None if config['devset'] is None else CoQADataset(config['devset'], config)
    test_set = None if config['testset'] is None else CoQADataset(config['testset'], config)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}

################################################################################
# Dataset Classes #
################################################################################


class CoQADataset(Dataset):
    """CoQA dataset."""

    def __init__(self, filename, config, training=False):
        timer = Timer('Load %s' % filename)
        self.filename = filename
        self.config = config
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        self.batch_size = config['batch_size']
        dataset = read_json(filename)
        for paragraph in dataset['data']:
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                n_history = len(history) if config['n_history'] < 0 else min(config['n_history'], len(history))
                if config['split_history']:
                    history_temp = []
                    if n_history > 0:
                        for i, (q, a) in enumerate(history[-n_history:]):
                            history_i = list()
                            history_i.append('<Q>')
                            history_i.extend(q)
                            history_i.append('<A>')
                            history_i.extend(a)
                            history_temp.append(history_i)
                    qas['annotated_history'] = history_temp
                temp.extend(qas['annotated_question']['word'])
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word']))
                qas['annotated_question']['word'] = temp
                self.examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                if config['split_history']:
                    for s in qas['annotated_history']:
                        for w in s:
                            self.vocab[w] += 1
                for w in qas['annotated_question']['word']:
                    self.vocab[w] += 1
                for w in paragraph['annotated_context']['word']:
                    self.vocab[w] += 1
                for w in qas['annotated_answer']['word']:
                    self.vocab[w] += 1
            self.paragraphs.append(paragraph)
        # Due to some unknown reasons (maybe pytorch bugs), if you try to use DataParallel(), the following code is
        # necessary. Otherwise, the model fails to work and tends to suddenly terminate after each epoch.
        if training and config['use_multi_gpu']:
            batch_num = len(self.examples) // self.batch_size
            self.examples = self.examples[0:(batch_num * self.batch_size)]
        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))
        print('Paragraph length: avg = %.1f, max = %d' % (np.average(paragraph_lens), np.max(paragraph_lens)))
        print('Question length: avg = %.1f, max = %d' % (np.average(question_lens), np.max(question_lens)))
        timer.finish()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qas = self.examples[idx]
        paragraph = self.paragraphs[qas['paragraph_id']]
        history = qas['annotated_history']
        question = qas['annotated_question']
        answers = [qas['answer']]
        if 'additional_answers' in qas:
            answers = answers + qas['additional_answers']

        sample = {'id': (paragraph['id'], qas['turn_id']),
                  'question': question,
                  'history': history,
                  'answers': answers,
                  'evidence': paragraph['annotated_context'],
                  'targets': None}
        if 'answer_span' in qas:
            sample['targets'] = qas['answer_span']

        return sample


################################################################################
# Read & Write Helper Functions #
################################################################################


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with io.open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def log_json(data, filename, mode='w', encoding='utf-8'):
    with io.open(filename, mode, encoding=encoding) as outfile:
        outfile.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_processed_file_contents(file_path, encoding="utf-8"):
    contents = get_file_contents(file_path, encoding=encoding)
    return contents.strip()

################################################################################
# DataLoader Helper Functions #
################################################################################


def sanitize_input(sample_batch, config, vocab):
    """
    Reformats sample_batch for easy vectorization.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        config: the configuration.
        vocab: word embedding dictionary.
    """
    sanitized_batch = defaultdict(list)
    for ex in sample_batch:
        history = ex['history']
        question = ex['question']['word']
        evidence = ex['evidence']['word']

        processed_q, processed_e, processed_h = [], [], []
        if config['split_history']:
            for s in history:
                processed_s = []
                for w in s:
                    processed_s.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
                processed_h.append(processed_s)
        for w in question:
            processed_q.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
        for w in evidence:
            processed_e.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])

        # Append relevant index-structures to batch
        sanitized_batch['history'].append(processed_h)
        sanitized_batch['question'].append(processed_q)
        sanitized_batch['evidence'].append(processed_e)
        sanitized_batch['history_text'].append(history)
        sanitized_batch['evidence_text'].append(evidence)
        sanitized_batch['question_text'].append(question)

        # featurize evidence document:
        sanitized_batch['targets'].append(ex['targets'])
        sanitized_batch['answers'].append(ex['answers'])
        if 'id' in ex:
            sanitized_batch['id'].append(ex['id'])
    return sanitized_batch


def vectorize_input(batch, config, char_vocab, training=True, device=None):
    """
    - Vectorize question and question mask
    - Vectorize evidence documents and mask
    - Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['question'])

    # Initialize all relevant parameters to None:
    max_word_len = config['max_word_length']

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for q in batch['question']])
    xq_mask = torch.ByteTensor(batch_size, max_q_len).fill_(1)
    for i, q in enumerate(batch['question']):
        xq_mask[i, :len(q)].fill_(0)

    # Part 2: Document Words
    max_d_len = max([len(d) for d in batch['evidence']])
    xd_mask = torch.ByteTensor(batch_size, max_d_len).fill_(1)
    for i, d in enumerate(batch['evidence']):
        xd_mask[i, :len(d)].fill_(0)

    # Part 3: History Words
    history_length = []
    turn_length = []
    for s in batch['history']:
        turn_length.append(len(s))
        for h in s:
            history_length.append(len(h))
    max_h_len = max(history_length)
    max_t_len = max(turn_length)
    xh_mask = torch.ByteTensor(batch_size, max_t_len, max_h_len).fill_(1)
    xh_t_mask = torch.ByteTensor(batch_size, max_t_len).fill_(1)
    for i, s in enumerate(batch['history']):
        xh_t_mask[i, :len(s)].fill_(0)
        for j, h in enumerate(s):
            xh_mask[i, j, :len(h)].fill_(0)

    # part 3: Char ids
    if not config['use_elmo']:
        xq = torch.LongTensor(batch_size, max_q_len).fill_(0)
        for i, q in enumerate(batch['question']):
            xq[i, :len(q)].copy_(torch.LongTensor(q))

        xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
        for i, d in enumerate(batch['evidence']):
            xd[i, :len(d)].copy_(torch.LongTensor(d))

        xh = torch.LongTensor(batch_size, max_t_len, max_h_len).fill_(0)
        for i, s in enumerate(batch['history']):
            for j, h in enumerate(s):
                xd[i, j, :len(h)].copy_(torch.LongTensor(h))

        xq_char = torch.LongTensor(batch_size, max_q_len, max_word_len).fill_(0)
        for i, sent in enumerate(batch['question_text']):
            for j, word in enumerate(sent):
                word = word.lower()
                char_range = (len(word) if len(word) < max_word_len else max_word_len)
                processed_char = []
                for k in range(char_range):
                    processed_char.append(char_vocab[word[k]] if word[k] in char_vocab else char_vocab[Constants._UNK_CHAR])
                xq_char[i, j, :char_range].copy_(torch.LongTensor(processed_char))

        xd_char = torch.LongTensor(batch_size, max_d_len, max_word_len).fill_(0)
        for i, sent in enumerate(batch['evidence_text']):
            for j, word in enumerate(sent):
                word = word.lower()
                char_range = (len(word) if len(word) < max_word_len else max_word_len)
                processed_char = []
                for k in range(char_range):
                    processed_char.append(char_vocab[word[k]] if word[k] in char_vocab else char_vocab[Constants._UNK_CHAR])
                xd_char[i, j, :char_range].copy_(torch.LongTensor(processed_char))

        xh_char = torch.LongTensor(batch_size, max_t_len, max_h_len, max_word_len).fill_(0)
        for i, turns in enumerate(batch['history_text']):
            for t, sent in enumerate(turns):
                for j, word in enumerate(sent):
                    word = word.lower()
                    char_range = (len(word) if len(word) < max_word_len else max_word_len)
                    processed_char = []
                    for k in range(char_range):
                        processed_char.append(char_vocab[word[k]] if word[k] in char_vocab else char_vocab[Constants._UNK_CHAR])
                    xh_char[i, t, j, :char_range].copy_(torch.LongTensor(processed_char))

    else:
        xq = batch_to_ids(batch['question_text'])
        xd = batch_to_ids(batch['evidence_text'])
        temp_xh = []
        for s in batch['history_text']:
            s.extend([[] for _ in range(max_t_len - len(s))])
            temp_xh.extend(s)
        xh = batch_to_ids(temp_xh).view(batch_size, max_t_len, max_h_len, 50)

    # Part 4: Target representations
    if not batch['targets'][0]:
        targets = None

    elif config['sum_loss']:  # For sum_loss "targets" acts as a mask rather than indices.
        targets = torch.ByteTensor(batch_size, max_d_len, 2).fill_(0)
        for i, _targets in enumerate(batch['targets']):
            for s, e in _targets:
                targets[i, s, 0] = 1
                targets[i, e, 1] = 1
    else:
        targets = torch.LongTensor(batch_size, 2)
        for i, _target in enumerate(batch['targets']):
            targets[i][0] = _target[0]
            targets[i][1] = _target[1]

    torch.set_grad_enabled(training)

    if not config['use_elmo']:
        example = {'batch_size': batch_size,
                   'answers': batch['answers'],
                   'xq': xq.to(device) if device else xq,
                   'xq_mask': xq_mask.to(device) if device else xq_mask,
                   'xq_char': xq_char.to(device) if device else xq_char,
                   'xh': xh.to(device) if device else xh,
                   'xh_mask': xh_mask.to(device) if device else xh_mask,
                   'xh_char': xh_char.to(device) if device else xh_char,
                   'xd': xd.to(device) if device else xd,
                   'xd_mask': xd_mask.to(device) if device else xd_mask,
                   'xd_char': xd_char.to(device) if device else xd_char,
                   'evidence_text': batch['evidence_text']}
    else:
        example = {'batch_size': batch_size,
                   'answers': batch['answers'],
                   'xq': xq.to(device) if device else xq,
                   'xq_mask': xq_mask.to(device) if device else xq_mask,
                   'xh': xh.to(device) if device else xh,
                   'xh_mask': xh_mask.to(device) if device else xh_mask,
                   'xd': xd.to(device) if device else xd,
                   'xd_mask': xd_mask.to(device) if device else xd_mask,
                   'evidence_text': batch['evidence_text']}
    if not targets is None:
        example['targets'] = targets.to(device) if device else targets

    return example
