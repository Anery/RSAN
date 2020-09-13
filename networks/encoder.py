import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        if self.opt.use_pos and self.opt.use_char:
            self.in_width = self.opt.word_embedding_size + self.opt.pos_embedding_size + self.opt.filter_number
        elif self.opt.use_pos:
            self.in_width = self.opt.word_embedding_size + self.opt.pos_embedding_size
        elif self.opt.use_char:
            self.in_width = self.opt.word_embedding_size + self.opt.filter_number
        else:
            self.in_width = self.opt.word_embedding_size
        self.birnn = nn.GRU(self.in_width, self.opt.rnn_hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.opt.rnn_hidden_size*2, self.opt.att_hidden_size)
        self.dropout = nn.Dropout(self.opt.dropout_rate)

    def forward(self, x, sen_len):
        sort_len, perm_idx = sen_len.sort(0, descending=True)
        _, un_idx = torch.sort(perm_idx, dim=0)
        x_input = x[perm_idx]
        packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)

        packed_out, _ = self.birnn(packed_input)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.opt.max_len)
        output = torch.index_select(output, 0, un_idx)
        output = torch.relu(self.linear(output))

        return output
