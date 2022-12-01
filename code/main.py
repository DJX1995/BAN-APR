import numpy as np
import time
import random
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import Prepare_logger
from evaluation import evaluate_1d
from BAN import BAN
from dataset import CharadesSTA, ActivityNet, TACoS, collate_fn, get_dataloader
from config import config_charades, config_anet, config_tacos
from tqdm import tqdm

# Set device
gpu_index = 0
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# if using cuda
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


def train_epoch(epoch, main_model, train_loader, optimizer, loss_weight, return_loss=False):
    '''
    :param epoch:
    :param main_model:
    :param train_loader:
    :param optimizer:
    :param mainloss:
    :param loss_weight: a dict storing weights for each loss value
    :param return_loss:
    :return:
    '''
    w1, w2, w3, w4, w5 = loss_weight['bce'], loss_weight['td'], loss_weight['refine'], \
                         loss_weight['contrast'], loss_weight['offset']
    main_model.train()
    optimizer.zero_grad()
    epoch_loss = []
    bce_loss = []
    refine_loss = []
    td_loss = []
    contrast_loss = []
    offset_loss = []

    start = time.time()

    for batch_idx, batch_data in tqdm(enumerate(train_loader), desc='training'):
        data, info = batch_data
        data = {key: value.to(device) for key, value in data.items()}
        out_feature, loss = main_model(data)
        total_loss = w1 * loss['loss_bce'] + w2 * loss['loss_td'] + w3 * loss['loss_refine'] + \
                     w4 * loss['loss_contrast'] + w5 * loss['loss_offset']
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(total_loss.item())
        bce_loss.append(loss['loss_bce'].item())
        refine_loss.append(loss['loss_refine'].item())
        td_loss.append(loss['loss_td'].item())
        contrast_loss.append(loss['loss_contrast'].item())
        offset_loss.append(loss['loss_offset'].item())

    end = time.time()
    time_cost = end - start
    losses = sum(epoch_loss) / len(epoch_loss)
    bce_loss = sum(bce_loss) / len(bce_loss)
    refine_loss = sum(refine_loss) / len(refine_loss)
    td_loss = sum(td_loss) / len(td_loss)
    contrast_loss = sum(contrast_loss) / len(contrast_loss)
    offset_loss = sum(offset_loss) / len(offset_loss)
    logger.info(
        f'Epoch {epoch}, Loss {losses:.5f}, bce_loss {bce_loss:.5f}, refine_loss {refine_loss:.5f}, '
        f'td_loss {td_loss:.5f}, contrast_loss {contrast_loss:.5f}, offset_loss {offset_loss:.5f},'
        f'time cost {time_cost / 60:.2f} minutes')

    if return_loss:
        return losses


def evaluate_epoch(epoch, main_model, data_loader, loss_weight, split='Validation'):
    w1, w2, w3, w4, w5 = loss_weight['bce'], loss_weight['td'], loss_weight['refine'], \
                             loss_weight['contrast'], loss_weight['offset']

    main_model.eval()
    with torch.no_grad():
        epoch_loss = []
        score_pred = []
        score_pred_1d = []
        prop_s_e = []
        time_stamp = []
        duration = []
        bce_loss = []
        refine_loss = []
        td_loss = []
        contrast_loss = []
        offset_loss = []
        num_clips = main_model.max_video_seq_len

        start = time.time()
        for batch_idx, batch_data in tqdm(enumerate(data_loader), desc=split):
            data, info = batch_data
            data = {key: value.to(device) for key, value in data.items()}
            out_feature, loss = main_model(data)
            # loss = mainloss(out_feature, data)
            total_loss = w1 * loss['loss_bce'] + w2 * loss['loss_td'] + w3 * loss['loss_refine'] + \
                         w4 * loss['loss_contrast'] + w5 * loss['loss_offset']
            tmap_pred = out_feature['tmap'].sigmoid_() * out_feature['map2d_mask']
            final_pred = out_feature['final_pred'].sigmoid()
            prop_s_e.append(out_feature['coarse_pred_round'])
            score_pred_1d.append(final_pred)

            score_pred.append(tmap_pred)
            time_stamp.append(data['timestamp'])
            duration.append(data['duration'])
            epoch_loss.append(total_loss.item())
            bce_loss.append(loss['loss_bce'].item())
            refine_loss.append(loss['loss_refine'].item())
            td_loss.append(loss['loss_td'].item())
            contrast_loss.append(loss['loss_contrast'].item())
            offset_loss.append(loss['loss_offset'].item())

        end = time.time()
        time_cost = end - start
        losses = sum(epoch_loss) / len(epoch_loss)
        bce_loss = sum(bce_loss) / len(bce_loss)
        refine_loss = sum(refine_loss) / len(refine_loss)
        td_loss = sum(td_loss) / len(td_loss)
        contrast_loss = sum(contrast_loss) / len(contrast_loss)
        offset_loss = sum(offset_loss) / len(offset_loss)

        score_pred = torch.cat(score_pred)
        time_stamp = torch.cat(time_stamp)
        duration = torch.cat(duration)
        prop_s_e = torch.cat(prop_s_e)
        score_pred_1d = torch.cat(score_pred_1d)

        recall_x_iou = evaluate_1d(score_pred_1d, prop_s_e, time_stamp, duration, num_clips=num_clips)
        logger.info(f'-----{split}-----')
        logger.info(f'Validation R1@IOU>0.3: {recall_x_iou[0, 0]:4f}, '
                    f'Validation R1@IOU>0.5: {recall_x_iou[0, 1]:4f}, '
                    f'Validation R1@IOU>0.7: {recall_x_iou[0, 2]:4f}, '
                    f'loss: Loss {losses:.10f}, bce_loss {bce_loss:.10f}, refine_loss {refine_loss:.10f}, '
                    f'td_loss {td_loss:.5f}, contrast_loss {contrast_loss:.10f}, offset_loss {offset_loss:.5f},'
                    f'time cost {time_cost / 60:.2f} minutes')
    return recall_x_iou


def evaluate_model(best_model, data_loader, config):
    # hyperparameters
    glove_emb_path = config.dataset.glove
    glove_emb = np.load(open(glove_emb_path, 'rb'))
    vocab_size = len(glove_emb)
    video_seq_len = config.model.video_seq_len

    main_model = BAN(vocab_size, config, pre_train_emb=glove_emb, device=device).to(device)
    main_model.load_state_dict(best_model)

    main_model.eval()
    score_pred = []
    score_pred_1d = []
    prop_s_e = []
    time_stamp = []
    duration = []
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            data, info = batch_data
            data = {key: value.to(device) for key, value in data.items()}
            out_feature, loss = main_model(data)
            tmap_pred = out_feature['tmap'].sigmoid_() * out_feature['map2d_mask']
            final_pred = out_feature['final_pred'].sigmoid()
            prop_s_e.append(out_feature['coarse_pred_round'])
            score_pred_1d.append(final_pred)

            score_pred.append(tmap_pred)
            time_stamp.append(data['timestamp'])
            duration.append(data['duration'])

    end = time.time()
    time_cost = end - start

    score_pred = torch.cat(score_pred)
    prop_s_e = torch.cat(prop_s_e)
    score_pred_1d = torch.cat(score_pred_1d)
    time_stamp = torch.cat(time_stamp)
    duration = torch.cat(duration)

    recall_x_iou = evaluate_1d(score_pred_1d, prop_s_e, time_stamp, duration, num_clips=video_seq_len)

    logger.info('-----Evaluation-----')
    logger.info(f'Validation R1@IOU>0.3: {recall_x_iou[0, 0]:4f}, '
                f'Validation R1@IOU>0.5: {recall_x_iou[0, 1]:4f}, '
                f'Validation R1@IOU>0.7: {recall_x_iou[0, 2]:4f}, '
                f'time cost {time_cost / 60:.2f} minutes')


def train(config):
    global logger  # , writer
    glove_emb_path = config.dataset.glove
    glove_emb = np.load(open(glove_emb_path, 'rb'))
    vocab_size = len(glove_emb)
    dataset_name = config.dataset.name
    data_type = config.dataset.datatype
    model_id = config.train.model_id

    # hyperparameters
    n_epoch = config.train.n_epoch
    batch_size = config.train.batch_size
    lr = config.train.lr
    decay_weight = config.train.decay_weight
    decay_step = config.train.decay_step
    video_seq_len = config.model.video_seq_len

    # logger
    logger = Prepare_logger('./logs/', print_console=False)

    train_loader, val_loader, test_loader = get_dataloader(config)
    main_model = BAN(vocab_size, config, glove_emb, device).to(device)

    if config.train.resume:
        load_model_path = f'../best_model/ban_{dataset_name}_{data_type}_{video_seq_len}_{model_id}.pth.tar'
        best_model = torch.load(load_model_path, map_location=device)
        main_model.load_state_dict(best_model)

    loss_weight = config.loss.loss_weight
    learned_params = filter(lambda p: p.requires_grad, main_model.parameters())
    optimizer = torch.optim.Adam(learned_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_weight)
    logger.info(
        f'ban model with {dataset_name} {data_type} features, video sequence length {video_seq_len}, '
        f'batch size {batch_size}, num_epochs {n_epoch}, learning rate {lr:.4f}, '
        f'proposal number {config.model.prop_num}')
    logger.info('loss weights: %s', loss_weight)
    start = time.time()
    dirname = f'../experiment/saved_models_id{model_id}'
    while os.path.exists(dirname):
        dirname = dirname + '_'
    os.makedirs(dirname)
    for epoch in range(n_epoch):
        epoch_loss = train_epoch(epoch, main_model, train_loader, optimizer, loss_weight, return_loss=True)
        if epoch >= config.test.start_epoch:
            if val_loader is not None:
                recall_x_iou_val = evaluate_epoch(epoch, main_model, val_loader, loss_weight, split='val')
            recall_x_iou_test = evaluate_epoch(epoch, main_model, test_loader, loss_weight, split='test')
            if config.train.save_best:
                torch.save(main_model.state_dict(),
                           f'{dirname}/ban_{dataset_name}_{data_type}_{video_seq_len}_id{model_id}_ep{epoch}.pth.tar')
        scheduler.step()
    end = time.time()
    print(f'total time cost: {(end - start) / 60:.2f} minutes')


def test(config, load_model_path):
    global logger  # , writer
    glove_emb_path = config.dataset.glove
    glove_emb = np.load(open(glove_emb_path, 'rb'))
    vocab_size = len(glove_emb)
    # logger
    logger = Prepare_logger('./logs/', print_console=False)

    train_loader, val_loader, test_loader = get_dataloader(config)

    main_model = BAN(vocab_size, config, glove_emb, device).to(device)
    best_model = torch.load(load_model_path, map_location=device)
    main_model.load_state_dict(best_model)

    evaluate_model(best_model, test_loader, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment args')
    parser.add_argument('--dataset', type=str, help='dataset name', nargs='?', default='charades')
    parser.add_argument('--run', type=str, help='train or test', nargs='?', default='train')
    parser.add_argument('--model-load-path', type=str, help='test model path', nargs='?',
                        default='../best_models/ban_anet_c3d_64_best.pth.tar')

    args = parser.parse_args()
    if args.dataset == 'anet':
        config = config_anet
    elif args.dataset == 'tacos':
        config = config_tacos
    else:
        config = config_charades
    if args.run == 'train':
        train(config=config)
    else:
        test(config, args.model_load_path)




