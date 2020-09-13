from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import random
from misc.utils import pickle_load

class Data(Dataset):

    def __init__(self, root, prefix):
        self.prefix = prefix
        self.features = np.load(os.path.join(root, prefix+'_features.npy'), allow_pickle=True)
        self.sen_len = np.load(os.path.join(root, prefix+'_len.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sen_len = self.sen_len[idx]
        feature = self.features[idx]
        return feature[0], feature[1], feature[2], feature[3], sen_len

class Train_data(Dataset):

    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(opt.root, 'train_pos_features.pkl'), 'rb') as f:
            self.features_pos = pickle_load(f)

        with open(os.path.join(opt.root, 'train_pos_len.pkl'), 'rb') as f:
            self.len_pos = pickle_load(f)

        with open(os.path.join(opt.root, 'train_neg_features.pkl'), 'rb') as f:
            self.features_neg = pickle_load(f)

        with open(os.path.join(opt.root, 'train_neg_len.pkl'), 'rb') as f:
            self.len_neg = pickle_load(f)

    def __len__(self):
        assert len(self.features_neg) == len(self.features_pos), 'length invalid'
        assert len(self.len_pos) == len(self.len_neg), 'length invalid'
        assert len(self.features_pos) == len(self.len_pos), 'length invalid'
        return len(self.len_pos)

    def __getitem__(self, idx):
        pos_f = self.features_pos[idx]
        pos_l = self.len_pos[idx]
        neg_f = self.features_neg[idx]
        neg_l = self.len_neg[idx]
        neg_zip = zip(neg_f, neg_l)
        #neg_num = int(len(neg_f)*self.opt.neg_rate)
        neg_num = self.opt.neg_num
        if neg_num != 0:
            neg_sam = random.sample(list(neg_zip), neg_num)
            neg_fs, neg_ls = zip(*neg_sam)
            example_f = pos_f + list(neg_fs)
            example_l = pos_l + list(neg_ls)
        else:
            example_f = pos_f
            example_l = pos_l
        sents, rels, labels, poses, chars = zip(*example_f)
        return sents, rels, labels, poses, chars, example_l

def dev_test_collate(features):
    sent = []
    triples = []
    poses = []
    chars = []
    sen_len = []
    for feature in features:
        sent.append(torch.tensor(feature[0]))
        poses.append(torch.tensor(feature[1]))
        chars.append(torch.tensor(feature[2]))
        triples.append(feature[3])
        sen_len.append(feature[4])
    sent = torch.stack(sent)
    poses = torch.stack(poses)
    chars = torch.stack(chars)
    sen_len = torch.tensor(sen_len)
    return sent, triples, poses, chars, sen_len

def train_collate(features):
    sent = []
    rel = []
    label = []
    pos = []
    chars = []
    sen_len = []
    for feature in features:
        sent.append(torch.tensor(feature[0]))
        rel.append(torch.tensor(feature[1]))
        label.append(torch.tensor(feature[2]))
        pos.append(torch.tensor(feature[3]))
        chars.append(torch.tensor(feature[4]))
        sen_len.append(torch.tensor(feature[5]))
    sent = torch.cat(sent, 0)
    rel = torch.cat(rel, 0)
    label = torch.cat(label, 0)
    pos = torch.cat(pos, 0)
    chars = torch.cat(chars, 0)
    sen_len = torch.cat(sen_len, 0)
    return sent, rel, label, pos, chars, sen_len

class Loader():
    def __init__(self, opt):
        self.opt = opt

        self.train_data = Train_data(opt)
        self.train_len = self.train_data.__len__()

        self.dev_data = Data(opt.root, 'dev')
        self.dev_len = self.dev_data.__len__()

        self.test_data = Data(opt.root, 'test')
        self.test_len = self.test_data.__len__()
        self.loader = {}
        self.reset('train')
        self.reset('dev')
        self.reset('test')


    def reset(self, prefix):
        if prefix == 'train':
            self.loader[prefix] = iter(DataLoader(self.train_data, batch_size=1, collate_fn=train_collate,
                                                   shuffle=True))

        if prefix == 'dev':
            self.loader[prefix] = iter(DataLoader(self.dev_data, batch_size=1, collate_fn=dev_test_collate, shuffle=False))

        if prefix == 'test':
            self.loader[prefix] = iter(DataLoader(self.test_data, batch_size=1, collate_fn=dev_test_collate, shuffle=False))

    def get_batch_train(self, batch_size):
        wrapped = False
        sents = []
        rels = []
        labels = []
        poses = []
        all_chars = []
        sen_lens = []
        for i in range(batch_size):
            try:
                sent, rel, label, pos, chars, sen_len = self.loader['train'].next()
            except:
                self.reset('train')
                sent, rel, label, pos, chars, sen_len = self.loader['train'].next()
                wrapped = True
            sents.append(sent)
            rels.append(rel)
            labels.append(label)
            poses.append(pos)
            all_chars.append(chars)
            sen_lens.append(sen_len)
        sents = torch.cat(sents, 0)
        rels = torch.cat(rels, 0)
        labels = torch.cat(labels, 0)
        poses = torch.cat(poses, 0)
        all_chars = torch.cat(all_chars, 0)
        sen_lens = torch.cat(sen_lens, 0)
        return sents, rels, labels, poses, all_chars, sen_lens, wrapped

    def get_batch_dev_test(self, batch_size, prefix):
        wrapped = False
        sents = []
        gts = []
        poses = []
        chars = []
        sen_lens = []
        for i in range(batch_size):
            try:
                sent, triple, pos, char, sen_len = self.loader[prefix].next()
            except:
                self.reset(prefix)
                sent, triple, pos, char, sen_len = self.loader[prefix].next()
                wrapped = True
            sents.append(sent[0])
            gts.append(triple[0])
            poses.append(pos[0])
            chars.append(char[0])
            sen_lens.append(sen_len[0])
        sents = torch.stack(sents)
        poses = torch.stack(poses)
        chars = torch.stack(chars)
        sen_lens = torch.stack(sen_lens)
        return sents, gts, poses, chars, sen_lens, wrapped

if __name__ =='__main__':
    import config
    opt = config.parse_opt()
    data_loader = Loader(opt)
    for i in range(1):
        data = data_loader.get_batch_train(1)
        #data = sorted(data, key=lambda x: list(x[-2].data), reverse=True)
        print(data[0].shape)
        print(data[1])
        print(data[2])
        print(data[3].shape)
        print(data[4].shape)
        print(data[5])

    print()

    for i in range(1):
        data = data_loader.get_batch_dev_test(10, 'dev')

        print(data[0].shape)
        print(data[1])
        print(data[2].shape)
        print(data[3].shape)
        print(data[4].shape)


