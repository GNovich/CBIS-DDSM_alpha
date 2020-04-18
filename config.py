from easydict import EasyDict as edict
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
from PIL import Image
import datetime
import time
import os

def get_config(training=True):
    conf = edict()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M')

    conf.data_path = Path('data')
    conf.work_path = Path('work_space')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log' / st
    conf.save_path = conf.work_path / 'save'
    conf.save_path = conf.work_path / 'save'
    conf.net_mode = 'resnet50'  # or 'ir

    conf.im_transform = trans.Compose([
            Image.fromarray,
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            trans.RandomRotation(30),  # subtle
            trans.Resize(225),
            trans.RandomCrop((225, 225)),
            trans.ToTensor()
    ])
    conf.data_mode = 'crop_data'
    conf.train_folder = conf.data_path / conf.data_mode / 'train'
    conf.test_folder = conf.data_path / conf.data_mode / 'test'
    conf.valid_ratio = .2
    conf.batch_size = 100  # irse net depth 50

    # --------------------Training Config ------------------------

    conf.lr = 1e-3
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 1
    conf.ce_loss = CrossEntropyLoss()

    return conf
