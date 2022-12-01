import numpy as np
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from dataset.anet import ActivityNet
from dataset.charades import CharadesSTA
from dataset.tacos import TACoS


def collate_fn(batch):
    visual, text, visual_len, iou2d, duration, timestamp, mask2d_contrast, \
    start_end_offset, start_end_gt, map_gt, vname, sentence = zip(*batch)
    text_lens = [len(t) for t in text]
    text_lens = torch.as_tensor(np.array(text_lens), dtype=torch.int64)
    video_len = [len(v) for v in visual]
    video_len = torch.as_tensor(np.array(video_len), dtype=torch.int64)
    visual = torch.stack(visual)
    iou2d = torch.stack(iou2d)
    s_e_distribution = torch.stack(map_gt)
    mask2d_contrast = torch.stack(mask2d_contrast)

    start_end_offset = torch.stack(start_end_offset)
    start_end_gt = torch.stack(start_end_gt)

    pad_text = pad_sequence(text, batch_first=True, padding_value=0)
    timestamp = torch.stack(timestamp)
    duration = torch.as_tensor(np.array(duration), dtype=torch.float)


    vname = list(vname)
    sentence = list(sentence)

    data = {'q_feature': pad_text,
            'q_len': text_lens,
            'v_feature': visual,
            'v_len': video_len,
            'timestamp': timestamp,
            'iou2d': iou2d,
            'start_end_offset': start_end_offset,
            'start_end_gt': start_end_gt,
            's_e_distribution': s_e_distribution,
            'duration': duration,
            'mask2d_contrast': mask2d_contrast
            }
    info = {'vname': vname,
            'sentence': sentence}
    return data, info


def get_dataloader(config):
    annotation_path = config.dataset.path.train_annotation
    annotation_path_val = config.dataset.path.val_annotation
    annotation_path_test = config.dataset.path.test_annotation
    word2idx_path = config.dataset.path.word2idx
    v_feat_path = config.dataset.path[config.dataset.datatype]
    v_feat_path_vgg = config.dataset.path.vgg
    dataset_name = config.dataset.name
    data_type = config.dataset.datatype
    num_workers = config.train.num_works
    test_batch_size = config.test.batch_size
    batch_size = config.train.batch_size
    video_seq_len = config.model.video_seq_len

    if dataset_name == 'charades':
        dataset_train = CharadesSTA(word2idx_path, annotation_path, v_feat_path, v_feat_path_vgg,
                                    max_video_seq_len=video_seq_len,
                                    subset=config.train.subset, data_type=data_type)
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_workers)
        dataset_test = CharadesSTA(word2idx_path, annotation_path_test, v_feat_path, v_feat_path_vgg,
                                   max_video_seq_len=video_seq_len,
                                   subset=config.test.subset, data_type=data_type)
        test_loader = DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=num_workers)
        val_loader = None
    elif dataset_name == 'anet':
        dataset_train = ActivityNet(word2idx_path, annotation_path, v_feat_path, v_feat_path_vgg,
                                    max_video_seq_len=video_seq_len,
                                    subset=config.train.subset)
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_workers)
        dataset_val = ActivityNet(word2idx_path, annotation_path_val, v_feat_path, v_feat_path_vgg,
                                  max_video_seq_len=video_seq_len, subset=config.test.subset)
        val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers)
        dataset_test = ActivityNet(word2idx_path, annotation_path_test, v_feat_path, v_feat_path_vgg,
                                   max_video_seq_len=video_seq_len,
                                   subset=config.test.subset)
        test_loader = DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=num_workers)
    else:
        dataset_train = TACoS(word2idx_path, annotation_path, v_feat_path, v_feat_path_vgg,
                                    max_video_seq_len=video_seq_len,
                                    subset=config.train.subset)
        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_workers)
        dataset_val = TACoS(word2idx_path, annotation_path_val, v_feat_path, v_feat_path_vgg,
                                  max_video_seq_len=video_seq_len, subset=config.test.subset)
        val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers)
        dataset_test = TACoS(word2idx_path, annotation_path_test, v_feat_path, v_feat_path_vgg,
                                   max_video_seq_len=video_seq_len,
                                   subset=config.test.subset)
        test_loader = DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=num_workers)
    return train_loader, val_loader, test_loader
