from easydict import EasyDict as edict
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
from PIL import Image

def get_config(training=True):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir
    conf.im_transform = trans.Compose([
            Image.fromarray,
            trans.Resize((225, 225)),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            # trans.Normalize([0.5], [0.5])
    ])
    conf.data_mode = 'crop'
    conf.train_folder = conf.data_path / 'train'
    conf.test_folder = conf.data_path / 'test'
    conf.valid_ratio = .2
    conf.batch_size = 100  # irse net depth 50

    # --------------------Training Config ------------------------

    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'
    conf.lr = 1e-3
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 3
    conf.ce_loss = CrossEntropyLoss()

    return conf
