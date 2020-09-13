import os
import torch
from .Rel_based_labeling import Rel_based_labeling

def setup(opt):
    if opt.model == 'Rel_based_labeling':
        Model = Rel_based_labeling(opt)
    else:
        raise Exception('Mode name error.')

    if opt.load_from is not None:
        assert os.path.isdir(opt.load_from), "%s must be a path" % opt.load_from
        Model.load_state_dict(torch.load(os.path.join(opt.load_from, 'model-best.pth')))

    return Model
