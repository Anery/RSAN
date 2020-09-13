from DataLoader import Loader
import os
import config
import model
import json
import eval_utils

opt = config.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

def load_label(input_label2id):
    label2id = json.load(open(input_label2id, 'r'))
    return label2id

def is_normal_triple(triples):
    """
    normal triples means triples are not over lap in entity.
    example [[e1,e2,r1], [e3,e4,r2]]
    :param triples
    :param is_relation_first
    :return:

    >>> is_normal_triple([1,2,3, 4,5,0])
    True
    >>> is_normal_triple([1,2,3, 4,5,3])
    True
    >>> is_normal_triple([1,2,3, 2,5,0])
    False
    >>> is_normal_triple([1,2,3, 1,2,0])
    False
    """
    entities = set()
    for e in triples:
        entities.add(e[0])
        entities.add(e[1])
    if len(entities) != 2 * len(triples):
        return False
    return True

def is_SEO(triples):
    if is_EPO(triples):
        return False
    if is_normal_triple(triples):
        return False
    return True

def is_EPO(triples):
    entity_pairs = set()
    for e in triples:
        e_pair = (e[0], e[1])
        entity_pairs.add(e_pair)
    if len(entity_pairs) != len(triples):
        return True
    return False

def overlapping_test(predictions, targets):
    normal_preds, normal_tar = [], []
    SEO_preds, SEO_tar = [], []
    EPO_preds, EPO_tar = [], []

    correct_preds_normal, total_preds_normal, total_gt_normal = 0., 0., 0.
    normal_num = 0

    correct_preds_SEO, total_preds_SEO, total_gt_SEO = 0., 0., 0.
    SEO_num = 0

    correct_preds_EPO, total_preds_EPO, total_gt_EPO = 0., 0., 0.
    EPO_num = 0

    for pred, tar in zip(predictions, targets):
        if is_normal_triple(tar):
            normal_preds.append(pred)
            normal_tar.append(tar)

            correct_preds_normal += len(set(pred) & set(tar))
            total_preds_normal += len(set(pred))
            total_gt_normal += len(set(tar))
            normal_num += 1
        elif is_SEO(tar):
            SEO_preds.append(pred)
            SEO_tar.append(tar)

            correct_preds_SEO += len(set(pred) & set(tar))
            total_preds_SEO += len(set(pred))
            total_gt_SEO += len(set(tar))
            SEO_num += 1
        elif is_EPO(tar):
            EPO_preds.append(pred)
            EPO_tar.append(tar)

            correct_preds_EPO += len(set(pred) & set(tar))
            total_preds_EPO += len(set(pred))
            total_gt_EPO += len(set(tar))
            EPO_num += 1

    normal_p, normal_r, normal_f = eval_utils.eval(correct_preds_normal, total_preds_normal, total_gt_normal)
    SEO_p, SEO_r, SEO_f = eval_utils.eval(correct_preds_SEO, total_preds_SEO, total_gt_SEO)
    EPO_p, EPO_r, EPO_f = eval_utils.eval(correct_preds_EPO, total_preds_EPO, total_gt_EPO)

    print('Normal metrics: ', normal_p, normal_r, normal_f)
    print('SEO metrics: ', SEO_p, SEO_r, SEO_f)
    print('EPO_num metrics: ', EPO_p, EPO_r, EPO_f)

def multiple_test(predictions, targets):
    preds1, tar1 = [], []
    preds2, tar2 = [], []
    preds3, tar3 = [], []
    preds4, tar4 = [], []
    preds5, tar5 = [], []

    correct_pred1, total_pred1, total_gt1 = 0., 0., 0.
    num1 = 0

    correct_pred2, total_pred2, total_gt2 = 0., 0., 0.
    num2 = 0
    
    correct_pred3, total_pred3, total_gt3 = 0., 0., 0.
    num3 = 0

    correct_pred4, total_pred4, total_gt4 = 0., 0., 0.
    num4 = 0

    correct_pred5, total_pred5, total_gt5 = 0., 0., 0.
    num5 = 0

    for pred, tar in zip(predictions, targets):
        l = len(tar)
        if l == 1:
            preds1.append(pred)
            tar1.append(tar)
            correct_pred1 += len(set(pred) & set(tar))
            total_pred1 += len(set(pred))
            total_gt1 += len(set(tar))
            num1 += 1
        if l == 2:
            preds2.append(pred)
            tar2.append(tar)
            correct_pred2 += len(set(pred) & set(tar))
            total_pred2 += len(set(pred))
            total_gt2 += len(set(tar))
            num2 += 1
        if l == 3:
            preds3.append(pred)
            tar3.append(tar)
            correct_pred3 += len(set(pred) & set(tar))
            total_pred3 += len(set(pred))
            total_gt3 += len(set(tar))
            num3 += 1
        if l == 4:
            preds4.append(pred)
            tar4.append(tar)
            correct_pred4 += len(set(pred) & set(tar))
            total_pred4 += len(set(pred))
            total_gt4 += len(set(tar))
            num4 += 1
        if l >= 5:
            preds5.append(pred)
            tar5.append(tar)    
            correct_pred5 += len(set(pred) & set(tar))
            total_pred5 += len(set(pred))
            total_gt5 += len(set(tar))
            num5 += 1 

    p1, r1, f1 = eval_utils.eval(correct_pred1, total_pred1, total_gt1)
    p2, r2, f2 = eval_utils.eval(correct_pred2, total_pred2, total_gt2)
    p3, r3, f3 = eval_utils.eval(correct_pred3, total_pred3, total_gt3)
    p4, r4, f4 = eval_utils.eval(correct_pred4, total_pred4, total_gt4)
    p5, r5, f5 = eval_utils.eval(correct_pred5, total_pred5, total_gt5)

    print('1 metrics: ', p1, r1, f1)
    print('2 metrics: ', p2, r2, f2)
    print('3 metrics: ', p3, r3, f3)
    print('4 metrics: ', p4, r4, f4)
    print('5 metrics: ', p5, r5, f5)

def Test(opt):
    loader = Loader(opt)
    Model = model.setup(opt).cuda()
    label2id = load_label(opt.input_label2id)
    predictions, targets, attention_score, metrics = eval_utils.evaluate(Model, loader, label2id, opt.eval_batch_size, opt.rel_num, 'test')
    rel2id = json.load(open(opt.input_rel2id, 'r'))
    id2rel = {v:k for k,v in rel2id.items()}

    overlapping_test(predictions, targets)
    multiple_test(predictions, targets)

    '''
    if opt.dump_results > 0:
        with open(os.path.join(opt.dump_path, 'good_results.json'), 'w') as w, open('data/no-ner-multi/test_sent_words', 'r') as f:
            test_sent = f.readlines()
            sent_attn_pairs = []
            for i, (pred,tar) in enumerate(zip(predictions, targets)):
                for j in range(len(attention_score[i])):
                    sent_attn_pair = {}
                    text = test_sent[i].strip().split('\t')
                    text.append(id2rel[j+1])
                    sent_attn_pair["text"] = text
                    sent_attn_pair["label"] = 0
                    sent_attn_pair["prediction"] = 0
                    sent_attn_pair["attention"] = attention_score[i][j]
                    sent_attn_pair["id"] = str(i) + '-' + str(j)
                    sent_attn_pairs.append(sent_attn_pair)
                if set(pred) == set(tar):
                    w.write(str(i) + '\n')
                    w.write(test_sent[i])
                    w.write('predict:' + '\n')
                    w.write(str(pred) + '\n')

                    w.write('target:' + '\n')
                    w.write(str(tar) + '\n')
                    w.write('\n')
                if i > 300:
                    break
            w.write('overall metrics: ' + str(metrics))
            print(len(sent_attn_pairs))
            json.dump(sent_attn_pairs, open('data/no-ner-multi/attention.json','w'), indent=1)
        #json.dump({'predictions': predictions, 'target': targets, 'metrics': metrics}, open(os.path.join(opt.dump_path, 'results.json'), 'w'))
    '''
Test(opt)
