import yaml
from easydict import EasyDict as edict

config = edict()

# dataset related
config.dataset = edict()
config.dataset.name = 'anet'
config.dataset.datatype = 'c3d'
config.dataset.ROOT = ''
config.dataset.glove = '../ActivityNet/data/caption/anet_caption_glove_embeds.npy'

config.dataset.path = edict()
config.dataset.path.c3d = '../ActivityNet/data/C3D_feature/c3d_fc6_stride_1s.hdf5'
config.dataset.path.vgg = ''
config.dataset.path.train_annotation = '../ActivityNet/data/caption/anet_caption_train.pkl'
config.dataset.path.val_annotation = '../ActivityNet/data/caption/anet_caption_val.pkl'
config.dataset.path.test_annotation = '../ActivityNet/data/caption/anet_caption_test.pkl'
config.dataset.path.word2idx = '../ActivityNet/data/caption/word2id.json'


video_seq_len = 64
neighbor = 3
topk = 20
negative = 0
prop_num = topk * (neighbor + 1) + negative   # 85

config.gcn = edict()
config.gcn.hidden_size = 512
config.gcn.k = prop_num
config.gcn.num_blocks = 2

# model related params
config.model = edict()
config.model.topk = topk
config.model.neighbor = neighbor
config.model.negative = negative
config.model.prop_num = prop_num
config.model.hidden_dim = 256
config.model.query_embed_dim = 300
config.model.visual_embed_dim = 4096  # 500
config.model.video_seq_len = video_seq_len
config.model.lstm_layer = 2
config.model.fuse_dim = 512
config.model.contrast_dim = 128
config.model.pooling_counts = [15, 8, 8]
config.model.drop_rate = 0.1
config.model.sparse_sample = True

# train
config.train = edict()
config.train.subset = None
config.train.lr = 0.001
config.train.decay_weight = 0.1
config.train.patient = 2
config.stage_epoch = 0
config.train.decay_step = 4
config.train.n_epoch = 8
config.train.batch_size = 32
config.train.resume = False
config.train.save_best = True
config.train.model_id = '0'

config.loss = edict()
config.loss.NAME = 'bce_loss'
config.loss.min_iou = 0.5
config.loss.max_iou = 1.0
config.loss.loss_weight = dict(zip(['bce', 'td', 'refine', 'contrast', 'offset'],
                                   [2., 0.1, 3., 0.1, 3.]))

# test
config.test = edict()
config.test.start_epoch = 2
config.test.subset = None
config.test.batch_size = 32


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))