import torch
from torch import nn
from rc.models import module
from torch.nn import functional as F


class RNet(nn.Module):
    def __init__(self, config, w_embedding):
        super(RNet, self).__init__()

        self.config = config
        self.hidden_size = hidden_size = config['hidden_size']
        self.w_embedding = w_embedding
        input_w_dim = self.w_embedding.embedding_dim

        if self.config['fix_embeddings']:
            for p in self.w_embedding.parameters():
                p.requires_grad = False

        self.char_encoder = module.CharEncoder(char_emb_size=config['embed_size'],
                                               hidden_size=hidden_size,
                                               dropout=config['dropout_emb'])
        q_input_size = input_w_dim + hidden_size * 2

        p_input_size = input_w_dim + hidden_size * 2

        self.sentence_encoder = module.SentenceEncoder(q_input_size=q_input_size,
                                                       p_input_size=p_input_size,
                                                       hidden_size=hidden_size,
                                                       num_layers=config['sent_rnn_layers'],
                                                       dropout=config['dropout_rnn'])

        if config['use_dot_attention']:
            self.pair_encoder = module.DotPairEncoder(
                p_input_size=hidden_size * 2 * config['sent_rnn_layers'],
                q_input_size=hidden_size * 2 * config['sent_rnn_layers'],
                hidden_size=hidden_size,
                dropout=config['dropout_rnn']
            )
        else:
            self.pair_encoder = module.PairEncoder(
                p_input_size=hidden_size * 2 * config['sent_rnn_layers'],
                q_input_size=hidden_size * 2 * config['sent_rnn_layers'],
                hidden_size=hidden_size,
                dropout=config['dropout_rnn']
            )

        if config['use_dot_attention']:
            self.self_match_encoder = module.DotSelfMatchEncoder(
                input_size=hidden_size * 2,
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
        q_emb = self.w_embedding(ex['xq'].transpose(0, 1))  # (batch, max_q_len, word_embed)
        p_emb = self.w_embedding(ex['xd'].transpose(0, 1))  # (batch, max_d_len, word_embed)
        q_emb = F.dropout(q_emb, self.config['dropout_emb'], self.training)
        p_emb = F.dropout(p_emb, self.config['dropout_emb'], self.training)

        q_char = ex['xq_char'].transpose(0, 1)
        p_char = ex['xd_char'].transpose(0, 1)
        p_mask = ex['xd_mask'].transpose(0, 1)
        q_mask = ex['xq_mask'].transpose(0, 1)

        q_char = self.char_encoder(q_char)
        p_char = self.char_encoder(p_char)
        question, passage = self.sentence_encoder(torch.cat((q_emb, q_char), dim=2), torch.cat((p_emb, p_char), dim=2))
        passage = self.pair_encoder(question, passage, q_mask)
        passage = self.self_match_encoder(passage, p_mask)
        pre = self.pointer_net(question, passage, q_mask, p_mask)

        return {'score_s': pre[0].transpose(0, 1),
                'score_e': pre[1].transpose(0, 1),
                'targets': ex['targets']}
