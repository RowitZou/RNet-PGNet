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
            q_input_size = p_input_size = h_input_size = 512

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

        q_encode_size = p_encode_size = h_encode_size = hidden_size * 2 * config['sent_rnn_layers']

        """
        self.q_h_encoder = module.DotAttentionEncoder(
            input_size=h_encode_size,
            hidden_size=h_encode_size // 2,
            dropout=config['dropout_rnn'],
            mem_1_size=q_encode_size
        )
        
        self.h_h_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2,
            memory_size=hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        self.h_q_encoder_att = module.DotAttentionEncoder(
            input_size=q_encode_size,
            memory_size=h_encode_size + hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )

        self.h_q_encoder_routing = module.DotRoutingEncoder(
            input_size=q_encode_size,
            memory_size=h_encode_size + hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn']
        )
        """
        self.q_p_encoder = module.DotAttentionEncoder(
            input_size=p_encode_size,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn'],
            mem_1_size=q_encode_size,
            mem_2_size=h_encode_size
        )

        self.p_p_encoder = module.DotAttentionEncoder(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn'],
            mem_1_size=hidden_size * 2
        )

        self.pointer_net = module.OutputLayer(
            q_input_size=q_encode_size,
            p_input_size=hidden_size * 2,
            hidden_size=hidden_size,
            dropout=config['dropout_rnn'],
            h_input_size=h_encode_size
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
        # q_h = self.q_h_encoder(history, question, q_mask)
        # h_h = self.h_h_encoder(q_h, q_h, h_mask)
        # h_q_att = self.h_q_encoder_att(torch.cat([history, q_h], dim=-1), question, h_mask)
        # print(self.h_q_encoder_att.attention[0])
        # h_q_routing = self.h_q_encoder_routing(torch.cat([history, h_h], dim=-1), question, h_mask, q_mask)
        # print(self.h_q_encoder_routing.attention[0])
        q_p = self.q_p_encoder(passage, question, q_mask, history, h_mask)
        p_p = self.p_p_encoder(q_p, q_p, p_mask)
        pre = self.pointer_net(question, p_p, q_mask, p_mask, history, h_mask)

        return {'score_s': pre[0].transpose(0, 1),
                'score_e': pre[1].transpose(0, 1),
                'targets': ex['targets']}
