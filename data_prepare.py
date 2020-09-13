#encoding=utf-8
import config
import json
import nltk
import os
import numpy as np
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


class DataPrepare(object):

    def __init__(self, opt):
        self.opt = opt
        vocab = np.load(opt.input_vocab)
        self.word2id = {j: i for i, j in enumerate(vocab)}
        self.id2word = {i: j for i, j in enumerate(vocab)}
        self.rel2id = json.load(open(opt.input_rel2id, 'r'))
        self.label2id = json.load(open(opt.input_label2id, 'r'))
        self.pos2id = json.load(open(opt.input_pos2id, 'r'))
        self.char2id = json.load(open(opt.input_char2id, 'r'))
        self.train_data = self.read_json(opt.input_train)
        self.test_data = self.read_json(opt.input_test)
        self.dev_data = self.read_json(opt.input_dev)

    def prepare(self):
        print('loading data ...')

        train_pos_f, train_pos_l, train_neg_f, train_neg_l = self.process_train(self.train_data)
        with open(os.path.join(''+self.opt.root, 'train_pos_features.pkl'), 'wb') as f:
            pickle_dump(train_pos_f, f)
        with open(os.path.join(''+self.opt.root, 'train_pos_len.pkl'), 'wb') as f:
            pickle_dump(train_pos_l, f)
        with open(os.path.join('' + self.opt.root, 'train_neg_features.pkl'), 'wb') as f:
            pickle_dump(train_neg_f, f)
        with open(os.path.join('' + self.opt.root, 'train_neg_len.pkl'), 'wb') as f:
            pickle_dump(train_neg_l, f)
        print('finish')

        dev_f, dev_l = self.process_dev_test(self.dev_data)
        np.save(os.path.join(''+self.opt.root, 'dev_features.npy'), dev_f, allow_pickle=True)
        np.save(os.path.join(''+self.opt.root, 'dev_len.npy'), dev_l, allow_pickle=True)

        test_f, test_l = self.process_dev_test(self.test_data)
        np.save(os.path.join('' + self.opt.root, 'test_features.npy'), test_f, allow_pickle=True)
        np.save(os.path.join('' + self.opt.root, 'test_len.npy'), test_l, allow_pickle=True)

    def read_json(self, filename):
        data = []
        with open('' + filename, 'r') as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def find_pos(self, sent_list, word_list):
        '''
        return position list
        '''
        l = len(word_list)
        for i in range(len(sent_list)):
            flag = True
            j = 0
            while j < l:
                if word_list[j] != sent_list[i + j]:
                    flag = False
                    break
                j += 1
            if flag:
                return range(i, i+l)
        return []

    def process_dev_test(self, dataset):
        features = []
        sen_len = []
        for i, data in enumerate(dataset):
            sent_text = data['sentText']
            sent_words, sent_ids, pos_ids, sent_chars, cur_len = self.process_sentence(sent_text)
            entities = data['entityMentions']
            raw_triples_ = data['relationMentions']
            # 去重
            triples_list = []
            for t in raw_triples_:
                triples_list.append((t['em1Text'], t['em2Text'], t['label']))
            triples_ = list(set(triples_list))
            triples_.sort(key=triples_list.index)

            triples = []
            for triple in triples_:
                head, tail, relation = triple
                try:
                    if triple[2] != 'None':
                        head_words = nltk.word_tokenize(head + ',')[:-1]
                        head_pos = self.find_pos(sent_words, head_words)
                        tail_words = nltk.word_tokenize(tail + ',')[:-1]
                        tail_pos = self.find_pos(sent_words, tail_words)
                        h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                        t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                        triples.append((h_chunk, t_chunk, self.rel2id[relation]))
                except:
                    continue


            features.append([sent_ids, pos_ids, sent_chars, triples])
            sen_len.append(cur_len)
            if (i + 1) * 1.0 % 10000 == 0:
                print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))
        return np.array(features), np.array(sen_len)

    def process_train(self, dataset):
        positive_features = []
        positive_lens = []
        negative_features = []
        negative_lens = []
        c = 0
        for i, data in enumerate(dataset):
            positive_feature = []
            positive_len = []
            negative_feature = []
            negative_len = []
            sent_text = data['sentText']
            # sent_chars : (max_len, max_word_len)
            sent_words, sent_ids, pos_ids, sent_chars, cur_len = self.process_sentence(sent_text)
            entities_ = data['entityMentions']
            entities = []
            for e_ in entities_:
                entities.append(e_['text'])

            raw_triples_ = data['relationMentions']
            # 去重
            triples_list = []
            for t in raw_triples_:
                triples_list.append((t['em1Text'], t['em2Text'], t['label']))
            triples_ = list(set(triples_list))
            triples_.sort(key=triples_list.index)

            triples = []
            cur_relations_list = []
            cur_relations_list.append(0)
            for triple in triples_:
                cur_relations_list.append(self.rel2id[triple[2]])
                head, tail, relation = triple
                try:
                    if triple[2] != 'None':
                        head_words = nltk.word_tokenize(head + ',')[:-1]
                        head_pos = self.find_pos(sent_words, head_words)
                        tail_words = nltk.word_tokenize(tail + ',')[:-1]
                        tail_pos = self.find_pos(sent_words, tail_words)
                        h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                        t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                        triples.append((h_chunk, t_chunk, self.rel2id[relation]))
                except:
                    continue

            cur_relations = list(set(cur_relations_list))
            cur_relations.sort(key=cur_relations_list.index)

            if len(cur_relations) == 1 and cur_relations[0] == 0:
                continue
            c += 1
            none_label = ['O'] * cur_len + ['X'] * (self.opt.max_len - cur_len)
            all_labels = {} #['O'] * self.max_len

            for triple in triples_:
                head, tail, relation = triple
                rel_id = self.rel2id[relation]
                #cur_label = none_label.copy()
                cur_label = all_labels.get(rel_id, none_label.copy())
                if triple[2] != 'None':
                    #label head
                    head_words = nltk.word_tokenize(head + ',')[:-1]
                    head_pos = self.find_pos(sent_words, head_words)
                    try:
                        if len(head_pos) == 1:
                            cur_label[head_pos[0]] = 'S-H'
                        elif len(head_pos) >= 2:
                            cur_label[head_pos[0]] = 'B-H'
                            cur_label[head_pos[-1]] = 'E-H'
                            for ii in range(1, len(head_pos)-1):
                                cur_label[head_pos[ii]] = 'I-H'
                    except:
                        continue

                    #label tail
                    tail_words = nltk.word_tokenize(tail + ',')[:-1]
                    tail_pos = self.find_pos(sent_words, tail_words)
                    try:
                        # not overlap enntity
                        if len(tail_pos) == 1:
                            cur_label[tail_pos[0]] = 'S-T'
                        elif len(tail_pos) >= 2:
                            cur_label[tail_pos[0]] = 'B-T'
                            cur_label[tail_pos[-1]] = 'E-T'
                            for ii in range(1, len(tail_pos)-1):
                                cur_label[tail_pos[ii]] = 'I-T'

                    except:
                        continue
                    all_labels[rel_id] = cur_label
            for ii in all_labels.keys():
                cur_label_ids = [self.label2id[e] for e in all_labels[ii]]
                positive_feature.append([sent_ids, ii, cur_label_ids, pos_ids, sent_chars])
                #positive_triple.append()
                positive_len.append(cur_len)

            none_label_ids = [self.label2id[e] for e in none_label]
            for r_id in range(self.opt.rel_num):
                if r_id not in cur_relations:
                    negative_feature.append([sent_ids, r_id, none_label_ids, pos_ids, sent_chars])
                    negative_len.append(cur_len)
            if (i + 1) * 1.0 % 10000 == 0:
                print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))
            positive_features.append(positive_feature)
            positive_lens.append(positive_len)
            negative_features.append(negative_feature)
            negative_lens.append(negative_len)
        print(c)

        return positive_features, positive_lens, negative_features, negative_lens


    def process_sentence(self, sent_text):
        sent_words = nltk.word_tokenize(sent_text)
        sen_len = min(len(sent_words), self.opt.max_len)
        sent_pos = nltk.pos_tag(sent_words)
        sent_pos_ids = [self.pos2id.get(pos[1], 1) for pos in sent_pos][:sen_len]
        sent_ids = [self.word2id.get(w, 1) for w in sent_words][:sen_len]

        sent_chars = []
        for w in sent_words[:sen_len]:
            tokens = [self.char2id.get(token, 1) for token in list(w)]
            word_len = min(len(tokens), self.opt.max_word_len)
            for _ in range(self.opt.max_word_len - word_len):
                tokens.append(0)
            sent_chars.append(tokens[: self.opt.max_word_len])

        for _ in range(sen_len, self.opt.max_len):
            sent_ids.append(0)
            sent_pos_ids.append(0)
            sent_chars.append([0] * self.opt.max_word_len)
        return sent_words[:sen_len], sent_ids, sent_pos_ids, sent_chars, sen_len

opt = config.parse_opt()
Prepare = DataPrepare(opt)
Prepare.prepare()
