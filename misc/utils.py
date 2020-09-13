#encoding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import six
from six.moves import cPickle

def pickle_load(f):
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)

def pickle_dump(obj, f):
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def build_optimizer(params, opt):
    if opt.optimizer == 'adam':
        return optim.Adam(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay, nesterov=True)
    else:
        raise Exception('optimizer is invalid.')

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, mask):
        loss_total = -input.gather(dim=2, index=target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss_total)
        return loss

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    #tag_type = content[1]
    if len(content) == 1:
        return tag_class
    ht = content[-1]
    return tag_class, ht


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    #print(seq)
    #print(tags)
    default1 = tags['O']
    default2 = tags['X']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if (tok == default1 or tok == default2) and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1 and tok != default2:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def attn_mapping(attn_scores, gts):
    gt_rel = []
    for gt in gts:
        gt_rel.append(gt[2] - 1)
    return attn_scores[gt_rel]

def tag_mapping(predict_tags, cur_relation, label2id):
    '''
    parameters
        predict_tags : np.array, shape: (rel_number, max_sen_len)
        cur_relation : list of relation id
    '''
    assert predict_tags.shape[0] == len(cur_relation)

    predict_triples = []
    for i in range(predict_tags.shape[0]):
        heads = []
        tails = []
        pred_chunks = get_chunks(predict_tags[i], label2id)
        for ch in pred_chunks:
            if ch[0].split('-')[-1] == 'H':
                heads.append(ch)
            elif ch[0].split('-')[-1] == 'T':
                tails.append(ch)
        #if heads.qsize() == tails.qsize():
        # TODO：当前策略：同等匹配，若头尾数量不符则舍弃多出来的部分
      
        if len(heads) != 0 and len(tails) != 0:
            if len(heads) < len(tails):
                heads += [heads[-1]] * (len(tails) - len(heads))
            if len(heads) > len(tails):
                tails += [tails[-1]] * (len(heads) - len(tails))
    
        for h_t in zip(heads, tails):
            #print(h_t)
            ht = list(h_t) + [cur_relation[i]]
            ht = tuple(ht)
            predict_triples.append(ht)
    return predict_triples
