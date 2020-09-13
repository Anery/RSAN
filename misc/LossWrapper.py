import torch
import torch.nn as nn
from misc.utils import CrossEntropy

class LossWrapper(nn.Module):
    def __init__(self, Model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = Model
        self.criterion = CrossEntropy()

    def forward(self, sent, sen_len, rel, mask, label, mask2, poses=None, chars=None):
        predict, weight = self.model(sent, sen_len, rel, mask, poses, chars)
        predict = torch.log_softmax(predict, dim=-1)

        loss = self.criterion(predict, label, mask2)
        return loss
