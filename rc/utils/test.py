# -*- coding: utf8 -*-

import torch
from rc.models.qp_encoder import QPEncoder
from rc.utils.argsParser import get_args
from rc.models.gated_attention_rnn import GatedAttentionRNN
from rc.models.self_matching_layer import SelfMatchingLayer
from rc.models.output_layer import OutputLayer


if __name__ == "__main__":

    args = get_args()
    passage_tensor = torch.randint(args.word_vocab_size, [10, 50], dtype=torch.int64).cuda()
    question_tensor = torch.randint(args.word_vocab_size, [10, 50], dtype=torch.int64).cuda()
    passage_tensor_char = torch.randint(args.char_vocab_size, [10, 50, 10], dtype=torch.int64).cuda()
    question_tensor_char = torch.randint(args.char_vocab_size, [10, 50, 10], dtype=torch.int64).cuda()

    qp_encoder = QPEncoder(args).cuda()
    gated_att_rnn = GatedAttentionRNN(args).cuda()
    self_matching_layer = SelfMatchingLayer(args).cuda()
    output_layer = OutputLayer(args).cuda()

    question_u, passage_u = qp_encoder(question_tensor_char, question_tensor, passage_tensor_char, passage_tensor)
    v_p = gated_att_rnn(question_u, passage_u)
    h_p = self_matching_layer(v_p)
    pre = output_layer(question_u, h_p)

    print(question_u.size())
    print(v_p.size())
    print(h_p.size())
    print(pre.size())
