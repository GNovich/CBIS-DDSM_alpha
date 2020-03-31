from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import numpy as np


def STL10Wrapper(**kwargs):
    if kwargs['train']:
        split_str = 'train'
    else:
        split_str = 'test'

    del (kwargs['train'])

    kwargs['split'] = split_str

    return datasets.STL10(**kwargs)


def SVHNWrapper(**kwargs):
    if kwargs['train']:
        split_str = 'train'
    else:
        split_str = 'test'

    del (kwargs['train'])

    kwargs['split'] = split_str

    return datasets.SVHN(**kwargs)


datasets_dict = {
    'MNIST': {
        'dataset': datasets.MNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),
                                 (0.3081,))
        ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (1, 28, 28),
        'output_dim': 10
    },
    'Fashion-MNIST': {
        'dataset': datasets.FashionMNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,),
                                 (0.3530,))
        ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (1, 28, 28),
        'output_dim': 10
    },
    'EMNIST': {
        'dataset': datasets.EMNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,),
                                 (0.3332,))
        ]),
        'dset_kwargs': {'split': 'balanced'},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (1, 28, 28),
        'output_dim': 47
    },
    'CIFAR-10': {
        'dataset': datasets.CIFAR10,
        'train_transform': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (3, 32, 32),
        'output_dim': 10
    },
    'CIFAR-100': {
        'dataset': datasets.CIFAR100,
        'train_transform': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))
        ]),
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))
        ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (3, 32, 32),
        'output_dim': 100
    },
    'SVHN': {
        'dataset': SVHNWrapper,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ]),
        'dset_kwargs': {},
        'val_size': 10000,
        'distribution': 'categorical',
        'input_shape': (3, 32, 32),
        'output_dim': 10
    },
    'STL10': {
        'dataset': STL10Wrapper,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2603, 0.2566, 0.2713))
        ]),
        'dset_kwargs': {},
        'val_size': 1000,
        'distribution': 'categorical',
        'input_shape': (3, 96, 96),
        'output_dim': 10
    }
}


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def load_morph_data(batch_size, dataset, depth):
    data_path = '/mnt/md0/orville/gleifman/CFAR_code/imgs_morph_new/' + dataset + '/' + str(depth)
    # data_path = '/mnt/md0/orville/Miriam/morph_neg/' + dataset + '/' + str(depth)
    if dataset == 'CIFAR-10':
        train_dataset = datasets.ImageFolder(
            root=data_path,
            # loader=default_loader,
            # extensions='.jpg',
            transform=transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
            ]))
    elif dataset == 'CIFAR-100':
        train_dataset = datasets.ImageFolder(
            root=data_path,
            # loader=default_loader,
            # extensions='.jpg',
            transform=transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762))
            ]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False

    )
    return train_loader


def get_dataloaders(batch_size, trial_i, dataset='MNIST', augment=False, early_stop=False, use_morph=False, depth=None,
                    n_workers=0):
    data_dir = './data/{}'.format(dataset)

    params = datasets_dict[dataset]

    datasets = {}
    for split in ['train', 'valid', 'test']:
        if augment and split == 'train' and 'train_transform' in params.keys():
            transform = params['train_transform']
        else:
            transform = params['transform']

        dset = params['dataset'](root=data_dir,
                                 train=(split != 'test'),
                                 download=True,
                                 transform=transform,
                                 **params['dset_kwargs'])
        datasets[split] = dset

    # Deterministic train/val split based on trial number
    if True:
        indices = list(range(len(datasets['train'])))
        val_size = params['val_size']

        s = np.random.RandomState(trial_i)
        valid_idx = s.choice(indices, size=val_size, replace=False)
        train_idx = list(set(indices) - set(valid_idx))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

    default_dloader_args = {
        'batch_size': batch_size,
        'pin_memory': True,
        'num_workers': n_workers,
        'drop_last': True,
        'shuffle': True
    }

    dataloaders = {}

    # If we're not doing early stopping, don't use a separate validation set
    if early_stop:
        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                          sampler=train_sampler,
                                          **default_dloader_args)
        dataloaders['valid'] = DataLoader(dataset=datasets['valid'],
                                          sampler=valid_sampler,
                                          **default_dloader_args)
        dataloaders['test'] = DataLoader(dataset=datasets['test'],
                                         shuffle=False,
                                         **default_dloader_args)
        dataloaders['morph'] = None
    else:
        if use_morph:
            train_morph_loader = load_morph_data(batch_size, dataset, depth)
            dataloaders['morph'] = train_morph_loader
        else:
            dataloaders['morph'] = None

        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                          shuffle=True,
                                          **default_dloader_args)
        dataloaders['valid'] = DataLoader(dataset=datasets['test'],
                                          shuffle=False,
                                          **default_dloader_args)
        dataloaders['test'] = DataLoader(dataset=datasets['test'],
                                         shuffle=False,
                                         **default_dloader_args)

    return dataloaders, params
