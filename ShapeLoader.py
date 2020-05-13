from functools import partial
from PIL import Image
from torchvision import transforms as trans
from torch.utils.data import Dataset
import numpy as np
from skimage.draw import random_shapes
import random
from itertools import product


class ShapeDataSet(Dataset):
    def __init__(self, prob_bkg=.5, no_bkg=False, aug=False, n_shapes=2, n_colors=2,
                 shape_only=False, color_only=False, im_size=224, ds_size=1e4):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.ds_size = int(ds_size)
        self.bkg_label = 0
        self.im_size = im_size
        self.prob_bkg = 0 if no_bkg else prob_bkg
        self.im_gen = partial(random_shapes, image_shape=(self.im_size, self.im_size), max_shapes=1,
                              multichannel=True, min_size=self.im_size // 4, max_size=(self.im_size * 9) // 4)
        self.null_im = Image.fromarray(np.ones(shape=(self.im_size, self.im_size, 3), dtype=np.uint8) * 255)
        self.null_im = trans.ToTensor()(self.null_im)
        self.aug = aug

        self.n_shapes = n_shapes
        assert n_shapes in [1, 2, 3]
        self.n_colors = n_colors
        assert n_colors in [1, 2, 3]

        self.shapes = ['rectangle', 'circle', 'triangle'][:n_shapes]
        self.colors = [[(255, 255), (0, 0), (0, 0)],
                       [(0, 0), (255, 255), (0, 0)],
                       [(0, 0), (0, 0), (255, 255)]]
        np.random.shuffle(self.colors)
        self.colors = self.colors[:n_colors]

        ziped_classes = enumerate(product(range(n_shapes), range(n_colors)))
        if shape_only:
            self.label_map = {v: v[0] + (1 - int(no_bkg)) for k, v in ziped_classes}
            self.label_names = [str(x) for x in self.shapes]
        elif color_only:
            self.label_map = {v: v[1] + (1 - int(no_bkg)) for k, v in ziped_classes}
            self.label_names = [str(x) for x in range(self.n_colors)]
        else:
            self.label_map = {v: k + (1 - int(no_bkg)) for k, v in ziped_classes}
            self.label_names = [str(x) for x in product(self.shapes, range(self.n_colors))]

        """
        class AddGaussianNoise(object):
            def __init__(self, mean=0., std=1.):
                self.std = std
                self.mean = mean

            def __call__(self, tensor):
                return tensor + (torch.randn(tensor.size()) * self.std) + self.mean

            def __repr__(self):
                return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        """

        self.transform = trans.Compose([
            Image.fromarray,
            trans.Pad(self.im_size, fill=(255, 255, 255)),
            trans.RandomAffine((-90, 90)),
            trans.CenterCrop(self.im_size),
            trans.ToTensor(),
            # AddGaussianNoise(0., .005)
        ])
        self.set_mode()  # default is train

    def gen_im(self, index):
        random.seed(index)
        if (random.random() < self.prob_bkg):
            im = self.null_im
            label = self.bkg_label
            return im, label
        else:
            shape_bit = random.choice(range(self.n_shapes))
            shape = self.shapes[shape_bit]
            color_bit = random.choice(range(self.n_colors))
            color = self.colors[color_bit]
            im = self.im_gen(shape=shape, intensity_range=color, random_seed=self.seed + index)[0]
            label = self.label_map[(shape_bit, color_bit)]
            return trans.ToTensor()(Image.fromarray(im)) if (not self.aug and shape_bit) else self.transform(im), label

    def set_mode(self, mode='train'):
        self.seed = 0 if mode == 'train' else self.ds_size + 1

    def __getitem__(self, index):
        if index >= self.ds_size:
            raise IndexError
        return self.gen_im(index)

    def __len__(self):
        return self.ds_size