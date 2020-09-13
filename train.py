import torch
import numpy as np
from DataLoader import Loader
import os
import config
import misc.utils as utils
import model
import traceback
import time
import json
from misc.LossWrapper import LossWrapper
import eval_utils

opt = config.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

def load_label(input_label2id):
    label2id = json.load(open(input_label2id, 'r'))
    return label2id


def save_checkpoint(model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
    torch.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos%s.pkl' %(append)), 'wb') as f:
        utils.pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(histories, f)

def train(opt):
    loader = Loader(opt)
    infos = {}
    histories = {}

    Model = model.setup(opt).cuda()
    LW_model = LossWrapper(Model, opt)
    # DP_lw_model = torch.nn.DataParallel(LW_model)
    LW_model.train()
    optimizer = utils.build_optimizer(Model.parameters(), opt)

    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos-best.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)

        if os.path.isfile(os.path.join(opt.start_from, 'histories-best.pkl')):
            with open(os.path.join(opt.start_from, 'histories-best.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)

        if os.path.isfile(os.path.join(opt.start_from, 'optimizer-best.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-best.pth')))
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['opt'] = opt
        infos['label2id'] = load_label(opt.input_label2id)

    iteration = infos.get('iter', '0')
    epoch = infos.get('epoch', '0')
    best_val_score = infos.get('best_val_score', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    epoch_done = True
    best_epoch = -1
    try:
        while True:
            if epoch_done:
                iteration = 0
                if epoch != 0:
                    predictions, targets, _ ,metrics = eval_utils.evaluate(Model, loader, infos['label2id'], opt.eval_batch_size, opt.rel_num, 'dev')
                    val_result_history[iteration] = {'predictions': predictions, 'metrics': metrics, 'targets': targets}
                    #print('dev res: ', metrics)
                    current_score = metrics['F1']
                    histories['c'] = val_result_history
                    histories['loss_history'] = loss_history
                    histories['lr_history'] = lr_history

                    best_flag = False
                    if current_score > best_val_score:
                        best_epoch = epoch
                        best_val_score = current_score
                        best_flag = True
                    infos['best_val_score'] = best_val_score

                    save_checkpoint(Model, infos, optimizer, histories)
                    if best_flag:
                        save_checkpoint(Model, infos, optimizer, append='best')


                epoch_done = False
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr)
            start = time.time()
            data = loader.get_batch_train(opt.batch_size)
            #data = sorted(data, key=lambda x: x[-1], reverse=True)
            wrapped = data[-1]
            data = data[:-1]
            #print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()
            data = [t.cuda() for t in data]
            sents, rels, labels, poses, chars, sen_lens = data
            if not opt.use_char:
                chars = None
            if not opt.use_pos:
                poses = None
            mask = torch.zeros(sents.size()).cuda()
            for i in range(sents.size(0)):
                mask[i][:sen_lens[i]] = 1

            mask2 = torch.where(labels == 12, torch.ones_like(sents), torch.ones_like(sents)*10).cuda()
            mask2 = mask2.float() * mask.float()

            optimizer.zero_grad()
            sum_loss = LW_model(sents, sen_lens, rels, mask, labels, mask2, poses, chars)

            loss = sum_loss/sents.shape[0]
            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            if iteration % 200 == 0:
                end = time.time()
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))

            iteration += 1
            if wrapped:
                epoch += 1
                epoch_done = True
            infos['iter'] = iteration
            infos['epoch'] = epoch

            if iteration % opt.save_loss_every == 0:
                loss_history[iteration] = train_loss
                lr_history[iteration] = opt.current_lr
            if opt.max_epoch != -1 and epoch >= opt.max_epoch:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        save_checkpoint(Model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

if __name__ == '__main__':
    train(opt)
