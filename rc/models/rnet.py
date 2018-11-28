import torch
from torch import nn
from rc.models import module


class RNet(nn.Module):
    def __init__(self, config, word_model=None):
        super(RNet, self).__init__()

        self.config = config
        self.hidden_size = hidden_size = config['hidden_size']

        if config['use_elmo']:
            self.w_embedding = module.ElmoLayer(options_file=config['elmo_options'],
                                                weights_file=config['elmo_weights'],
                                                requires_grad=config['elmo_fine_tune'])
            # Elmo embedding size: 1024
            q_input_size = p_input_size = 1024

        else:
            self.w_embedding = module.WordEmbedLayer(vocab_size=word_model.vocab_size,
                                                     embed_size=config['embed_size'],
                                                     pretrained_vecs=word_model.get_word_vecs(),
                                                     fix_embeddings=config['fix_embeddings'])

            self.char_encoder = module.CharEncoder(char_emb_size=config['embed_size'],
                                                   hidden_size=hidden_size)

            input_w_dim = self.w_embedding.embedding_dim

            q_input_size = input_w_dim + hidden_size * 2

            p_input_size = input_w_dim + hidden_size * 2

        self.sentence_encoder = module.SentenceEncoder(q_input_size=q_input_size,
                                                       p_input_size=p_input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=config['sent_rnn_layers'],
                                                       dropout_rnn=config['dropout_rnn'],
                                                       dropout_embed=config['dropout_emb'])

        self.q_pair_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2 * config['sent_rnn_layers'],
            memory_size=hidden_size * 2 * config['sent_rnn_layers'],
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        if config['use_dot_attention']:
            self.p_pair_encoder = module.DotAttentionEncoder(
                input_size=hidden_size * 2 * config['sent_rnn_layers'],
                memory_size=hidden_size * 2 * config['sent_rnn_layers'] + hidden_size * 2,
                hidden_size=hidden_size,
                dropout=config['dropout_rnn']
            )
        else:
            self.p_pair_encoder = module.PairEncoder(
                p_input_size=hidden_size * 2 * config['sent_rnn_layers'],
                q_input_size=hidden_size * 2 * config['sent_rnn_layers'],
                hidden_size=hidden_size,
                dropout=config['dropout_rnn']
            )

        if config['use_dot_attention']:
            self.self_match_encoder = module.DotAttentionEncoder(
                input_size=hidden_size * 2,
                memory_size=hidden_size * 2,
                hidden_size=hidden_size,
                dropout=config['dropout_rnn']
            )
        else:
            self.self_match_encoder = module.SelfMatchEncoder(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                dropout=config['dropout_rnn']
            )

        self.pointer_net = module.OutputLayer(
            q_input_size=hidden_size * 2 * config['sent_rnn_layers'],
            p_input_size=hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

    def forward(self, ex):
        # Embed both document and question
        # ex['xd_mask']  (batch_size, p_seq_length) ByteTensor
        # ex['xq_mask']  (batch_size, q_seq_length) ByteTensor

        p_mask = ex['xd_mask'].transpose(0, 1)
        q_mask = ex['xq_mask'].transpose(0, 1)

        if self.config['use_elmo']:
            q_emb = self.w_embedding(ex['xq']).transpose(0, 1)
            p_emb = self.w_embedding(ex['xd']).transpose(0, 1)
        else:
            q_emb = self.w_embedding(ex['xq'].transpose(0, 1))
            p_emb = self.w_embedding(ex['xd'].transpose(0, 1))
            q_char = self.char_encoder(ex['xq_char'].transpose(0, 1))
            p_char = self.char_encoder(ex['xd_char'].transpose(0, 1))
            q_emb = torch.cat((q_emb, q_char), dim=-1)
            p_emb = torch.cat((p_emb, p_char), dim=-1)

        # q_emb     (q_seq_length, batch_size, elmo_emb_size)
        # p_emb     (p_seq_length, batch_size, elmo_emb_size)

        question, passage = self.sentence_encoder(q_emb, p_emb)
        question_atten = self.q_pair_encoder(passage, question, p_mask)
        passage = self.p_pair_encoder(torch.cat((question, question_atten), dim=-1), passage, q_mask)
        if self.config['use_dot_attention']:
            passage = self.self_match_encoder(passage, passage, p_mask)
        else:
            passage = self.self_match_encoder(passage, p_mask)
        pre = self.pointer_net(question, passage, q_mask, p_mask)

        return {'score_s': pre[0].transpose(0, 1),
                'score_e': pre[1].transpose(0, 1),
                'targets': ex['targets']}
