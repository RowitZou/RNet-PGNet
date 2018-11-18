import torch
from torch import nn
from torch.nn import functional as F
from rc.utils import constants as Constants


class Gate(nn.Module):
    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size, bias=False),
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
        mask = input.new_ones(1, input.size(1), input.size(2), requires_grad=False)
        return self.dropout(mask) * input


class CharEncoder(nn.Module):

    def __init__(self, char_emb_size, hidden_size, dropout):
        super(CharEncoder, self).__init__()
        self.dropout = dropout
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
        char_output = F.dropout(char_output, self.dropout, self.training)
        # output [passage_length, batch_size, char_hidden_size * 2]
        return char_output


class SentenceEncoder(nn.Module):
    def __init__(self, q_input_size, p_input_size, hidden_size, num_layers, dropout):
        super(SentenceEncoder, self).__init__()

        self.num_layers = num_layers
        self.question_encoder = nn.ModuleList([
            nn.Sequential(
                # nn.Dropout(dropout),
                nn.GRU(input_size=q_input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
            )])
        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.question_encoder.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.GRU(input_size=hidden_size * 2,
                               hidden_size=hidden_size,
                               bidirectional=True)
                    ))

        self.passage_encoder = nn.ModuleList([
            nn.Sequential(
                # nn.Dropout(dropout),
                nn.GRU(input_size=p_input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
            )])
        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.passage_encoder.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.GRU(input_size=hidden_size * 2,
                               hidden_size=hidden_size,
                               bidirectional=True)
                    ))

    def forward(self, question, passage):

        question_outputs = []
        passage_outputs = []
        q_hidden = question
        p_hidden = passage
        for i in range(self.num_layers):
            q_hidden, _ = self.question_encoder[i](q_hidden)
            question_outputs.append(q_hidden)

        for i in range(self.num_layers):
            p_hidden, _ = self.passage_encoder[i](p_hidden)
            passage_outputs.append(p_hidden)

        question_outputs = torch.cat(question_outputs, dim=-1)
        passage_outputs = torch.cat(passage_outputs, dim=-1)
        return question_outputs, passage_outputs


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


class DotPairEncoder(nn.Module):
    def __init__(self, p_input_size, q_input_size, hidden_size, dropout):
        super(DotPairEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(p_input_size, hidden_size, bias=False),
            nn.ReLU()
        )
        self.memory_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(q_input_size, hidden_size, bias=False),
            nn.ReLU()
        )
        self.gate = Gate(p_input_size + q_input_size)
        self.rnn = nn.GRU(input_size=p_input_size + q_input_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, q_input, p_input, q_mask):
        q_input = q_input.transpose(0, 1)
        p_input = p_input.transpose(0, 1)  # Turn into batch first.
        q_mask = q_mask.transpose(0, 1)  # Turn into batch first.

        input_ = self.input_linear(p_input)
        memory_ = self.memory_linear(q_input)

        # Compute scores
        scores = input_.bmm(memory_.transpose(2, 1)) / (self.hidden_size ** 0.5)  # (batch, len1, len2)

        # Mask padding
        memory_mask = q_mask.unsqueeze(1).expand(scores.size())  # (batch, len1, len2)
        scores.masked_fill_(memory_mask, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=-1)

        # Take weighted average
        matched_seq = alpha.bmm(q_input)
        rnn_input = torch.cat((p_input, matched_seq), dim=-1).transpose(0, 1)
        rnn_input = self.gate(rnn_input)

        self.rnn.flatten_parameters()
        output, _ = self.rnn(rnn_input)
        return output


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


class DotSelfMatchEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU()
        )
        self.memory_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=input_size * 2, hidden_size=hidden_size, bidirectional=True)

    def forward(self, p_input, p_mask):

        p_input = p_input.transpose(0, 1)       # Turn into batch first.
        p_mask = p_mask.transpose(0, 1)         # Turn into batch first.

        input_ = self.input_linear(p_input)
        memory_ = self.memory_linear(p_input)

        # Compute scores
        scores = input_.bmm(memory_.transpose(2, 1)) / (self.hidden_size ** 0.5)     # (batch, len1, len2)

        # Mask padding
        memory_mask = p_mask.unsqueeze(1).expand(scores.size())  # (batch, len1, len2)
        scores.masked_fill_(memory_mask, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=-1)

        # Take weighted average
        matched_seq = alpha.bmm(p_input)
        rnn_input = torch.cat((p_input, matched_seq), dim=-1).transpose(0, 1)

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

        for t in range(2):
            p_weights = p_temp + self.passage_w[1](state)
            p_weights = self.passage_linear(p_weights)
            p_weights.masked_fill_(p_mask.unsqueeze(2), -float('inf'))
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
