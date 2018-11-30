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
            q_input_size = p_input_size = h_input_size = 1024

        else:
            self.w_embedding = module.WordEmbedLayer(vocab_size=word_model.vocab_size,
                                                     embed_size=config['embed_size'],
                                                     pretrained_vecs=word_model.get_word_vecs(),
                                                     fix_embeddings=config['fix_embeddings'])

            self.char_encoder = module.CharEncoder(char_emb_size=config['embed_size'],
                                                   hidden_size=hidden_size)

            input_w_dim = self.w_embedding.embedding_dim

            q_input_size = p_input_size = h_input_size = input_w_dim + hidden_size * 2

        self.passage_encoder = module.SentenceEncoder(input_size=p_input_size,
                                                      hidden_size=hidden_size,
                                                      num_layers=config['sent_rnn_layers'],
                                                      dropout_rnn=config['dropout_rnn'],
                                                      dropout_embed=config['dropout_emb'])

        self.question_encoder = module.SentenceEncoder(input_size=q_input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=config['sent_rnn_layers'],
                                                       dropout_rnn=config['dropout_rnn'],
                                                       dropout_embed=config['dropout_emb'])

        self.history_encoder = module.SentenceEncoder(input_size=h_input_size,
                                                      hidden_size=hidden_size,
                                                      num_layers=config['sent_rnn_layers'],
                                                      dropout_rnn=config['dropout_rnn'],
                                                      dropout_embed=config['dropout_emb'])

        self.h_q_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2 * config['sent_rnn_layers'],
            memory_size=hidden_size * 2 * config['sent_rnn_layers'],
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        self.p_q_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2 * config['sent_rnn_layers'] + hidden_size * 2,
            memory_size=hidden_size * 2 * config['sent_rnn_layers'],
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        self.q_p_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2 * config['sent_rnn_layers'],
            memory_size=hidden_size * 2 * config['sent_rnn_layers'] + hidden_size * 4,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        self.self_match_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2,
            memory_size=hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        self.pointer_net = module.OutputLayer(
            q_input_size=hidden_size * 2 * config['sent_rnn_layers'] + hidden_size * 2,
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
        h_mask = ex['xh_mask'].transpose(0, 1)

        if self.config['use_elmo']:
            q_emb = self.w_embedding(ex['xq']).transpose(0, 1)
            p_emb = self.w_embedding(ex['xd']).transpose(0, 1)
            h_emb = self.w_embedding(ex['xh']).transpose(0, 1)
        else:
            q_emb = self.w_embedding(ex['xq'].transpose(0, 1))
            p_emb = self.w_embedding(ex['xd'].transpose(0, 1))
            h_emb = self.w_embedding(ex['xh'].transpose(0, 1))
            q_char = self.char_encoder(ex['xq_char'].transpose(0, 1))
            p_char = self.char_encoder(ex['xd_char'].transpose(0, 1))
            h_char = self.char_encoder(ex['xh_char'].transpose(0, 1))
            q_emb = torch.cat((q_emb, q_char), dim=-1)
            p_emb = torch.cat((p_emb, p_char), dim=-1)
            h_emb = torch.cat((h_emb, h_char), dim=-1)

        # q_emb     (q_seq_length, batch_size, elmo_emb_size)
        # p_emb     (p_seq_length, batch_size, elmo_emb_size)

        question = self.question_encoder(q_emb)
        passage = self.passage_encoder(p_emb)
        history = self.history_encoder(h_emb)
        h_q = self.h_q_encoder(history, question, h_mask)
        p_q = self.p_q_encoder(passage, torch.cat([question, h_q], dim=-1), p_mask)
        q_p = self.q_p_encoder(torch.cat([question, h_q, p_q], dim=-1), passage, q_mask)
        passage = self.self_match_encoder(q_p, q_p, p_mask)
        pre = self.pointer_net(torch.cat([question, h_q], dim=-1), passage, q_mask, p_mask)

        return {'score_s': pre[0].transpose(0, 1),
                'score_e': pre[1].transpose(0, 1),
                'targets': ex['targets']}
