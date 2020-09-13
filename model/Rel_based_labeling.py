import torch.nn as nn
from networks.embedding import *
from networks.encoder import *
from networks.decoder import *

class Rel_based_labeling(nn.Module):

    def __init__(self, opt):
        super(Rel_based_labeling, self).__init__()
        self.opt = opt
        self.embedding = Embedding(opt)
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, sent, sen_len, rel, mask, poses=None, chars=None):
        embedding = self.embedding(sent, poses, chars)
        sen_embedding = self.encoder(embedding, sen_len)
        return self.decoder(sen_embedding, rel, mask, sen_len)
