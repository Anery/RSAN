import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionDot(nn.Module):
    def __init__(self, opt):
        super(AttentionDot, self).__init__()
        self.opt = opt
        self.rel2att = nn.Linear(self.opt.rel_dim, self.opt.att_hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.rel2att.weight.data)

    def forward(self, sent, rel, mask):
        # sent: batch, max_len, hidden
        # rel: batch, rel_dim -> relation: batch, hidden
        relation = self.rel2att(rel).unsqueeze(1)
        # batch, max_len
        weight = torch.matmul(relation, sent.transpose(-1,-2)).squeeze()
        weight = weight * mask.float()
        weight = torch.softmax(weight, -1)
        att_res = torch.bmm(weight.unsqueeze(1), sent).squeeze(1) # batch_size * att_hidden_size
        return att_res, weight

class AttentionNet(nn.Module):
    def __init__(self, opt):
        super(AttentionNet, self).__init__()
        self.opt = opt
        self.Wg = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.Wh = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.Wr = nn.Linear(self.opt.rel_dim, self.opt.att_hidden_size)
        self.alpha_net = nn.Linear(self.opt.att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wg.weight.data)
        nn.init.xavier_uniform_(self.Wh.weight.data)
        nn.init.xavier_uniform_(self.Wr.weight.data)
        nn.init.xavier_uniform_(self.alpha_net.weight.data)

    def forward(self, sent_h, rel, pool, mask):
        relation = self.Wr(rel)
        sent = self.Wh(sent_h)
        global_sen = self.Wg(pool)

        relation = relation.unsqueeze(1).expand_as(sent)
        global_sen = global_sen.unsqueeze(1).expand_as(sent)

        mix = torch.tanh(relation + sent + global_sen)
        weight = self.alpha_net(mix).squeeze()
        weight.masked_fill_(mask==0, -1e9)
        weight_ = torch.softmax(weight, -1)

        #weight = weight * mask.float()
        att_res = torch.bmm(weight_.unsqueeze(1), sent).squeeze(1) # batch_size * att_hidden_size
        return att_res, weight_

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.relation_matrix = nn.Embedding(self.opt.rel_num, self.opt.rel_dim)

        self.attention = AttentionNet(opt)
        self.W = nn.Linear(self.opt.att_hidden_size*3, self.opt.att_hidden_size)
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        self.bilstm = nn.LSTM(self.opt.att_hidden_size, self.opt.rnn_hidden_size, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.opt.rnn_hidden_size*2, self.opt.label_num)

        self.W1 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.W2 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.W3 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size*2)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.xavier_uniform_(self.hidden2tag.weight.data)
        #nn.init.kaiming_uniform_(self.W.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.W.weight.data)
        nn.init.xavier_uniform_(self.W1.weight.data)
        nn.init.xavier_uniform_(self.W2.weight.data)
        nn.init.xavier_uniform_(self.W3.weight.data)

    def masked_mean(self, sent, mask):
        mask_ = mask.masked_fill(mask==0, -1e9)
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def nn_decode(self, inputs, sen_len):
        sort_len, perm_idx = sen_len.sort(0, descending=True)
        _, un_idx = torch.sort(perm_idx, dim=0)
        x_input = inputs[perm_idx]
        packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)

        packed_out, _ = self.bilstm(packed_input)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.opt.max_len)
        output = torch.index_select(output, 0, un_idx)
        return output

    def forward(self, sent, rel, mask, sen_len):
        rel_embedding = self.relation_matrix(rel)
        global_sen = self.masked_mean(sent, mask)
        sent_att, weight = self.attention(sent, rel_embedding, global_sen, mask)

        concats = torch.cat([self.W1(global_sen), self.W2(sent_att)], -1)
        alpha = torch.sigmoid(concats)
        gate = alpha * torch.tanh(self.W3(sent_att))
        decode_input = torch.cat([sent, gate.unsqueeze(1).expand(sent.shape[0], sent.shape[1], -1)], -1)
        decode_input = self.W(decode_input)
        #decode_input = sent + (alpha * (sum_sen_rel)).unsqueeze(1).expand_as(sent)
        decode_out = self.nn_decode(decode_input, sen_len)
        project = self.hidden2tag(decode_out)
        return project, weight
