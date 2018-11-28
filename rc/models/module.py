import torch
from torch import nn
from torch.nn import functional as F
from rc.utils import constants as Constants
from allennlp.modules.elmo import Elmo


class Gate(nn.Module):
    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return input * self.gate(input)


class RNNDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, input):

        if not self.training:
            return input
        mask = input.new_ones(input.size(0), 1, input.size(2), requires_grad=False)
        return self.dropout(mask) * input


class ElmoLayer(nn.Module):
    def __init__(self, options_file=None, weights_file=None, requires_grad=False):
        super(ElmoLayer, self).__init__()
        if not options_file:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
                           "2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        if not weights_file:
            weights_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
                           "2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

        self.elmo = Elmo(options_file, weights_file, 1, dropout=0, requires_grad=requires_grad)

    def forward(self, char_ids):
        embeddings = self.elmo(char_ids)
        return embeddings['elmo_representations'][0]


class WordEmbedLayer(nn.Module):

    def __init__(self, vocab_size, embed_size, pretrained_vecs=None, fix_embeddings=True):
        super(WordEmbedLayer, self).__init__()
        self.word_embed_layer = nn.Embedding(vocab_size, embed_size, padding_idx=0,
                                             _weight=torch.from_numpy(pretrained_vecs).float()
                                             if pretrained_vecs is not None else None)
        if fix_embeddings:
            for p in self.word_embed_layer.parameters():
                p.requires_grad = False

        self.embedding_dim = self.word_embed_layer.embedding_dim

    def forward(self, word_input):
        return self.word_embed_layer(word_input)


class CharEncoder(nn.Module):

    def __init__(self, char_emb_size, hidden_size):
        super(CharEncoder, self).__init__()
        self.char_embeddings = nn.Embedding(len(Constants._ALPHABETS) + 2, char_emb_size, padding_idx=0)

        # Here we employ bidirectional GRU to encode char-level embeddings. It can be replaced by CNN layer.
        self.biGRU_char = nn.GRU(char_emb_size, hidden_size, bidirectional=True)

    def forward(self, input_tensor):

        # input_tensor [passage_length, batch_size, word_length]
        passage_length = input_tensor.shape[0]
        batch_size = input_tensor.shape[1]

        char_embedding = self.char_embeddings(input_tensor.contiguous().view([batch_size * passage_length, -1])).transpose(0, 1)
        self.biGRU_char.flatten_parameters()
        char_output, _ = self.biGRU_char(char_embedding)
        char_output = char_output[len(char_output) - 1].contiguous().view([passage_length, batch_size, -1])

        # output [passage_length, batch_size, char_hidden_size * 2]
        return char_output


class SentenceEncoder(nn.Module):
    def __init__(self, q_input_size, p_input_size, hidden_size, num_layers, dropout_rnn, dropout_embed):
        super(SentenceEncoder, self).__init__()

        self.num_layers = num_layers
        self.question_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_embed),
                nn.GRU(input_size=q_input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
            )])
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.question_encoder.append(
                    nn.Sequential(
                        nn.Dropout(dropout_rnn),
                        nn.GRU(input_size=q_input_size + hidden_size * 2 * (i + 1),
                               hidden_size=hidden_size,
                               bidirectional=True)
                    ))

        self.passage_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_embed),
                nn.GRU(input_size=p_input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
            )])
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.passage_encoder.append(
                    nn.Sequential(
                        nn.Dropout(dropout_rnn),
                        nn.GRU(input_size=p_input_size + hidden_size * 2 * (i + 1),
                               hidden_size=hidden_size,
                               bidirectional=True)
                    ))

    def forward(self, question, passage):

        question_outputs = []
        passage_outputs = []
        question_inputs = []
        passage_inputs = []
        question_inputs.append(question)
        passage_inputs.append(passage)

        for i in range(self.num_layers):
            self.question_encoder[i][1].flatten_parameters()
            q_hidden, _ = self.question_encoder[i](torch.cat(question_inputs, dim=-1))
            question_outputs.append(q_hidden)
            question_inputs.append(q_hidden)

        for i in range(self.num_layers):
            self.passage_encoder[i][1].flatten_parameters()
            p_hidden, _ = self.passage_encoder[i](torch.cat(passage_inputs, dim=-1))
            passage_outputs.append(p_hidden)
            passage_inputs.append(p_hidden)

        question_outputs = torch.cat(question_outputs, dim=-1)
        passage_outputs = torch.cat(passage_outputs, dim=-1)
        return question_outputs, passage_outputs


class DotAttentionEncoder(nn.Module):
    def __init__(self, input_size, memory_size, hidden_size, dropout):
        super(DotAttentionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU()
        )
        self.memory_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(memory_size, hidden_size, bias=False),
            nn.ReLU()
        )
        self.gate = Gate(input_size + memory_size)
        self.rnn = nn.GRU(input_size=input_size + memory_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, memory_input, seq_input, memory_mask):
        memory_input = memory_input.transpose(0, 1)
        seq_input = seq_input.transpose(0, 1)  # Turn into batch first.
        memory_mask = memory_mask.transpose(0, 1)  # Turn into batch first.

        input_ = self.input_linear(seq_input)
        memory_ = self.memory_linear(memory_input)

        # Compute scores
        scores = input_.bmm(memory_.transpose(2, 1)) / (self.hidden_size ** 0.5)  # (batch, len1, len2)

        # Mask padding
        memory_mask = memory_mask.unsqueeze(1).expand(scores.size())  # (batch, len1, len2)
        scores.masked_fill_(memory_mask, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=-1)

        # Take weighted average
        matched_seq = alpha.bmm(memory_input)
        rnn_input = torch.cat((seq_input, matched_seq), dim=-1).transpose(0, 1)
        rnn_input = self.gate(rnn_input)

        self.rnn.flatten_parameters()
        output, _ = self.rnn(rnn_input)
        return output


class PairEncoderCell(nn.Module):
    def __init__(self, p_input_size, q_input_size, hidden_size, dropout):
        super(PairEncoderCell, self).__init__()

        self.GRUCell = nn.GRUCell(p_input_size + q_input_size, hidden_size)

        self.attention_w = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(q_input_size, hidden_size, bias=False)),
            nn.Sequential(nn.Dropout(dropout), nn.Linear(p_input_size, hidden_size, bias=False)),
            nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size, bias=False))
        ])

        self.attention_s = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

        self.gate = Gate(p_input_size + q_input_size)

    def forward(self, q_input, p_input, q_mask, state):
        # q_input: q_length * batch_size * hidden_size
        # p_input: batch_size * hidden_size
        # q_mask:  q_length * batch_size * hidden_size
        # state:   batch_size * hidden_size

        q_temp = self.attention_w[0](q_input)
        p_temp = self.attention_w[1](p_input)
        state_temp = self.attention_w[2](state)

        attention_logits = self.attention_s(q_temp + p_temp + state_temp)
        attention_logits.masked_fill_(q_mask.unsqueeze(2), -float('inf'))
        attention_weights = F.softmax(attention_logits, dim=0)
        attention_vec = torch.sum(attention_weights * q_input, dim=0)

        new_input = torch.cat([p_input, attention_vec], dim=-1)
        new_input = self.gate(new_input)

        return self.GRUCell(new_input, state)


class PairEncoder(nn.Module):
    def __init__(self, p_input_size, q_input_size, hidden_size, dropout):
        super(PairEncoder, self).__init__()
        self.forward_cell = PairEncoderCell(p_input_size, q_input_size, hidden_size, dropout)
        self.backward_cell = PairEncoderCell(p_input_size, q_input_size, hidden_size, dropout)
        self.hidden_size = hidden_size

    def forward(self, q_input, p_input, q_mask):

        output_fw, _ = self.unroll_attention(self.forward_cell, q_input, p_input, q_mask, backward=False)
        output_bw, _ = self.unroll_attention(self.backward_cell, q_input, p_input, q_mask, backward=True)

        return torch.cat([output_fw, output_bw], dim=-1)

    def unroll_attention(self, cell, q_input, p_input, q_mask, backward=False):
        output = []

        state = p_input.new_zeros(p_input.shape[1], self.hidden_size, requires_grad=False)
        if backward:
            steps = range(len(p_input)-1, -1, -1)
        else:
            steps = range(len(p_input))

        for t in steps:
            state = cell(q_input, p_input[t], q_mask, state)
            output.append(state)

        if backward:
            output = output[::-1]

        output = torch.stack(output, dim=0)
        return output, state


class SelfMatchEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(SelfMatchEncoder, self).__init__()

        self.attention_w = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(input_size, hidden_size, bias=False)),
            nn.Sequential(nn.Dropout(dropout), nn.Linear(input_size, hidden_size, bias=False))
        ])
        self.attention_s = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

        self.rnn = nn.GRU(input_size * 2, hidden_size, bidirectional=True)

    def forward(self, p_input, p_mask):

        # Avoid out-of-memory
        chunk_size = 100

        collection = []
        max_doc_length = len(p_input)
        chunk_num = max_doc_length // chunk_size
        remain_chunk_size = max_doc_length - (chunk_num * chunk_size)

        v_p_temp = self.attention_w[0](p_input)
        v_p_chunk_temp = self.attention_w[0](p_input)

        for chunk_i in range(chunk_num + 1):
            if chunk_i >= chunk_num and remain_chunk_size > 0:
                temp_chunk = v_p_chunk_temp[chunk_i * chunk_size: chunk_i * chunk_size + remain_chunk_size]
            elif chunk_i >= chunk_num and remain_chunk_size == 0:
                break
            else:
                temp_chunk = v_p_chunk_temp[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]
            attention_weights = v_p_temp + temp_chunk.unsqueeze(1).expand(-1, max_doc_length, -1, -1)
            attention_weights = self.attention_s(attention_weights)
            attention_weights.masked_fill_(p_mask.unsqueeze(2), -float('inf'))
            attention_logits = F.softmax(attention_weights, dim=1)
            attention_vecs = torch.sum(attention_logits * p_input, dim=1)
            collection.append(attention_vecs)

        rnn_input = torch.cat((torch.cat(collection, dim=0), p_input), dim=2)

        self.rnn.flatten_parameters()
        output, _ = self.rnn(rnn_input)

        return output


class OutputLayer(nn.Module):
    def __init__(self, q_input_size, p_input_size, hidden_size, dropout):
        super(OutputLayer, self).__init__()

        self.rnn_cell = nn.GRUCell(p_input_size, q_input_size)
        self.passage_w = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(p_input_size, hidden_size, bias=False)),
            nn.Sequential(nn.Dropout(dropout), nn.Linear(q_input_size, hidden_size, bias=False))
        ])
        self.passage_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.question_w = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(q_input_size, hidden_size, bias=False)),
            nn.Sequential(nn.Dropout(dropout), nn.Linear(q_input_size, hidden_size, bias=False))
        ])
        self.question_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.V_q = nn.Parameter(torch.randn(q_input_size), requires_grad=True)

    def forward(self, q_input, p_input, q_mask, p_mask):
        state_weights = self.question_w[0](q_input) + self.question_w[1](self.V_q)
        state_weights = self.question_linear(state_weights)
        state_weights.masked_fill_(q_mask.unsqueeze(2), -float('inf'))
        state_attention = F.softmax(state_weights, dim=0)
        state = torch.sum(state_attention * q_input, dim=0)

        pre = []
        p_temp = self.passage_w[0](p_input)

        for t in range(3):
            p_weights = p_temp + self.passage_w[1](state)
            p_weights = self.passage_linear(p_weights)
            p_weights.masked_fill_(p_mask.unsqueeze(2), -float('inf'))
            if t > 0:
                pre.append(F.log_softmax(p_weights.squeeze(), dim=0))
            p_attention = F.softmax(p_weights, dim=0)
            p_vecs = torch.sum(p_attention * p_input, dim=0)
            state = self.rnn_cell(p_vecs, state)

        pre = torch.stack(pre)
        return pre


################################################################################
# Functional #
################################################################################

def multi_nll_loss(scores, target_mask):
    """
    Select actions with sampling at train-time, argmax at test-time:
    """
    scores = scores.exp()
    loss = 0
    for i in range(scores.size(0)):
        loss += torch.neg(torch.log(torch.masked_select(scores[i], target_mask[i]).sum() / scores[i].sum()))
    return loss
