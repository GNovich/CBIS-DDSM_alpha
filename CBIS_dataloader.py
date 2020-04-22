import os
from functools import partial
import cv2
from PIL import Image
from torchvision import transforms as trans
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import pandas as pd
import numpy as np
import tqdm
from random

class CBIS_Dataloader:
    """
        A that yeilds labeled patch data given an mamograph and ROI image.
    """
    def __init__(self, n_patches, conf, og_resize=(1152,896), patch_size=224, roi_sampling_ratio=.5, seed=12345):
        # image file path, cropped image file path, ROI mask file pat
        self.roi_dir_name = 'data/ROI_file'
        self.mam_dir_name = 'data/image_file'
        self.crop_dir_name = 'data/crop_data'

        self.roi_sampling_ratio = roi_sampling_ratio
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.n_src_per_batch = max(conf.batch_size // self.n_patches, 1)
        self.og_resize = og_resize
        self.seed = seed

        csv_dir = 'csv_files'
        self.train_table = pd.concat([pd.read_csv(os.path.join(csv_dir, x)) for x in
                                     os.listdir(csv_dir) if 'train' in x])
        self.test_table = pd.concat([pd.read_csv(os.path.join(csv_dir, x)) for x in
                                     os.listdir(csv_dir) if 'test' in x])
        # adding label for pos patches, bkg patch is 0
        self.train_table['pos_label'] = (pd.Series(
            zip(self.train_table['label'], self.train_table['abnormality type'])).astype('category').cat.codes + 1).values.astype(int)
        self.test_table['pos_label'] = (pd.Series(
            zip(self.test_table['label'], self.test_table['abnormality type'])).astype('category').cat.codes + 1).values.astype(int)

        # 1 know faulty sample
        self.test_table = self.test_table[self.test_table['ROI mask file path png'] != 'Calc-Training_P_00474_LEFT_MLO_1.png']
        self.train_table = self.train_table[
            self.train_table['ROI mask file path png'] != 'Calc-Training_P_00474_LEFT_MLO_1.png']

        self.src_transform = trans.Compose([
            Image.fromarray,
            trans.Resize(og_resize),
            trans.Pad(patch_size // 2)
        ])

        class RightAngleTransform:
            """Rotate by one of the right angles."""

            def __init__(self):
                self.angles = [0, 90, 180, 270]

            def __call__(self, x):
                angle = np.random.choice(self.angles)
                return trans.functional.rotate(x, angle)

        self.patch_transform = trans.Compose([
            #trans.Normalize([.5, .5]),
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            RightAngleTransform(),
            trans.ToTensor()
        ])

        self.dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False,
            'shuffle': True
        }

        self.grey_loader = partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
        file_ext = ('.png',)
        self.dataset = DatasetFolder(conf.train_folder, extensions=file_ext,
                                     loader=self.grey_loader, transform=self.src_transform)
        self.train_loader = DataLoader(self.dataset, **self.dloader_args)

        self.test_ds = DatasetFolder(conf.test_folder, extensions=file_ext,
                                     loader=self.grey_loader, transform=self.src_transform)
        self.test_loader = DataLoader(self.test_ds, **self.dloader_args)

    def get_cont(self, im):
        _, contours, _ = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [cv2.contourArea(cont) for cont in contours]
        idx = np.argmax(cont_areas)  # find the largest contour.
        rx, ry, rw, rh = cv2.boundingRect(contours[idx])
        return rx, ry, rw, rh

        def sample_patches(self, img, roi_image, pos_cutoff=.9, neg_cutoff=.1, nb_abn=10, nb_bkg=10, hard_center=True):

        patch_size = self.patch_size
        rng = np.random.RandomState(self.seed)
        roi_f = (np.array(roi_image) > 0).astype(float)
        roi_size = roi_f.sum()
        patch_cover = cv2.blur(roi_f, (patch_size, patch_size))
        roi_cover = (patch_cover * (patch_size ** 2)) / roi_size

        # TODO maybe skip
        # edge case - might be less then thresh
        image_thresh = max(patch_cover.max(), roi_cover.max())
        pos_cutoff = image_thresh * .95 if image_thresh < pos_cutoff else pos_cutoff

        # abnormalities
        abn_filter = (patch_cover > pos_cutoff) | (roi_cover > pos_cutoff)
        abn_targets = np.argwhere(abn_filter)

        # force center in image and sample
        abn_targets = abn_targets[(abn_targets[:, 0] + (patch_size // 2) < roi_f.shape[0]) &
                                  (abn_targets[:, 0] - (patch_size // 2) > 0) &
                                  (abn_targets[:, 1] + (patch_size // 2) < roi_f.shape[1]) &
                                  (abn_targets[:, 1] - (patch_size // 2) > 0)]

        if hard_center:
            upleft_x, upleft_y, rw, rh = self.get_cont(np.array(roi_image))
            abn_targets = abn_targets[(abn_targets[:, 0] < upleft_y + rh) &
                                      (abn_targets[:, 0] > upleft_y) &
                                      (abn_targets[:, 1] < upleft_x + rw) &
                                      (abn_targets[:, 1] > upleft_x)]

        if len(abn_targets) < 1: return []  # TODO make sure no funny buisness
        abn_targets = abn_targets[rng.choice(len(abn_targets), nb_abn)]

        # background
        bkg_filter = (patch_cover < neg_cutoff) & (roi_cover < neg_cutoff)
        bkg_targets = np.argwhere(bkg_filter)
        # force center in image and sample
        bkg_targets = bkg_targets[(bkg_targets[:, 0] + (patch_size // 2) < roi_f.shape[0]) &
                                  (bkg_targets[:, 0] - (patch_size // 2) > 0) &
                                  (bkg_targets[:, 1] + (patch_size // 2) < roi_f.shape[1]) &
                                  (bkg_targets[:, 1] - (patch_size // 2) > 0)]
        if len(bkg_targets) < 1: return []  # TODO make sure no funny buisness
        bkg_targets = bkg_targets[rng.choice(len(bkg_targets), nb_bkg)]

        patches = []
        targets = np.concatenate([abn_targets, bkg_targets])
        for target_x, target_y in targets:
            patch = img.crop((target_y - patch_size // 2, target_x - patch_size // 2,
                              target_y + patch_size // 2, target_x + patch_size // 2))
            patches.append(self.patch_transform(patch))
        return patches

    def get_loader(self, mode='train', pos_cutoff=.9, neg_cutoff=.1, sample=1):
        roi_col = 'ROI mask file path png'
        mam_col = 'image file path png'

        nb_abn = int(self.n_patches * self.roi_sampling_ratio)
        nb_bkg = self.n_patches - nb_abn
        patch_batch = []
        patch_labels = []
        counter = 0
        table = (self.train_table if mode=='train' else self.test_table)
        for (og_i, row) in (table.sample(frac=sample).iterrows()):
            mam_path = os.path.join(self.mam_dir_name, mode, str(row['label']), row[mam_col])
            if not os.path.exists(mam_path):
                continue
            mam_image = self.src_transform(self.grey_loader(mam_path))
            roi_path = os.path.join(self.roi_dir_name, mode, str(row['label']), row[roi_col])
            if not os.path.exists(roi_path):
                continue
            roi_image = self.src_transform(self.grey_loader(roi_path))

            patches = self.sample_patches(mam_image, roi_image, nb_abn=nb_abn, nb_bkg=nb_bkg,
                                                   pos_cutoff=pos_cutoff, neg_cutoff=neg_cutoff)
            if len(patches) < 1: continue  # TODO make sure no funny buisness
            labels = ([row['pos_label']]*nb_abn) + ([0]*nb_bkg)
            patch_batch.extend(patches)
            patch_labels.extend(labels)
            counter += 1

            if counter % self.n_src_per_batch == 0:
                # yeild batch
                shuffle_idx = np.random.permutation(len(patch_labels))
                yield torch.stack(patch_batch)[shuffle_idx], torch.Tensor(patch_labels).long()[shuffle_idx]
                patch_batch = []
                patch_labels = []
                counter = 0

    @property
    def test_len(self):
        return len(self.test_table) // self.n_src_per_batch

    @property
    def train_len(self):
        return len(self.train_table) // self.n_src_per_batch


class SourceDat(Dataset):
    def __init__(self, mode='train', seed=None, og_resize=(1152,896), patch_size=224, nb_abn=10, nb_bkg=10):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.og_resize = og_resize
        self.patch_size = patch_size
        self.nb_abn = nb_abn
        self.nb_bkg = nb_bkg
        self.seed = seed
        csv_dir = 'csv_files'
        self.table = pd.concat([pd.read_csv(os.path.join(csv_dir, x)) for x in os.listdir(csv_dir) if mode in x])
        # adding label for pos patches, bkg patch is 0
        self.table['pos_label'] = (pd.Series(
            zip(self.table['label'], self.table['abnormality type'])).astype('category').cat.codes + 1).values

        # 1 know faulty sample
        self.table = self.table[self.table['ROI mask file path png'] != 'Calc-Training_P_00474_LEFT_MLO_1.png']

        roi_path_func = lambda row: os.path.join('data/ROI_file', mode, str(row['label']),
                                                 row['ROI mask file path png'])
        mam_path_func = lambda row: os.path.join('data/image_file', mode, str(row['label']), row['image file path png'])

        # image paths
        self.roi_paths = self.table.apply(roi_path_func, axis=1).values
        self.mam_paths = self.table.apply(mam_path_func, axis=1).values

        # abels
        self.label_arr = self.table['pos_label'].astype(int).values
        # Calculate len
        self.data_len = len(self.label_arr)

        self.grey_loader = partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
        self.transform = trans.Compose([
            Image.fromarray,
            trans.Resize(og_resize),
            trans.Pad(patch_size // 2),
            trans.ToTensor()
        ])

    def __getitem__(self, index):
        # Open image
        roi_im = self.transform(self.grey_loader(self.roi_paths[index]))
        mam_im = self.transform(self.grey_loader(self.mam_paths[index]))
        return (roi_im, mam_im, self.label_arr[index])

    def __len__(self):
        return self.data_len


class CBIS_PatchDataSet_INMEM(Dataset):
    def __init__(self, mode='train', seed=None, og_resize=(1152,896), patch_size=224, patch_num=10, prob_bkg=.5):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.og_resize = og_resize
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.prob_bkg = prob_bkg
        self.seed = seed
        csv_dir = 'csv_files'
        self.table = pd.concat([pd.read_csv(os.path.join(csv_dir, x)) for x in os.listdir(csv_dir) if mode in x])
        # 1 know faulty sample
        self.table = self.table[self.table['ROI mask file path png'] != 'Calc-Training_P_00474_LEFT_MLO_1.png']

        roi_path_func = lambda row: os.path.join('data/ROI_file', mode, str(row['label']),
                                                 row['ROI mask file path png'])
        mam_path_func = lambda row: os.path.join('data/image_file', mode, str(row['label']), row['image file path png'])

        # image paths
        self.roi_paths = self.table.apply(roi_path_func, axis=1).values
        self.mam_paths = self.table.apply(mam_path_func, axis=1).values

        self.grey_loader = partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
        self.transform = trans.Compose([
            Image.fromarray,
            trans.Resize(og_resize),
            trans.Pad(patch_size // 2),
        ])

        class RightAngleTransform:
            """Rotate by one of the right angles."""

            def __init__(self):
                self.angles = [0, 90, 180, 270]

            def __call__(self, x):
                angle = np.random.choice(self.angles)
                return trans.functional.rotate(x, angle)

        self.patch_transform = trans.Compose([
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            RightAngleTransform(),
            trans.ToTensor(),
            #trans.Normalize([.5], [.5])
        ])

        self.roi_im = []
        self.mam_im = []
        src_dat = SourceDat(mode=mode)

        dloader_args = {
            'batch_size': 10,
            'pin_memory': False,
            'num_workers': 20,
            'drop_last': False,
            'shuffle': False
        }
        self.label_arr = []
        src_loader = DataLoader(src_dat, **dloader_args)
        for roi, mam, label in tqdm.tqdm(src_loader, total=len(src_loader), desc='loading '+str(mode)):
            for i in range(len(roi)):
                self.mam_im.append(trans.ToPILImage()(mam[i]))
                self.roi_im.append(trans.ToPILImage()(roi[i]))
                self.label_arr.append(label[i].item())
                # Calculate len
        self.data_len = len(self.label_arr)

    def get_cont(self, im):
        _, contours, _ = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [cv2.contourArea(cont) for cont in contours]
        idx = np.argmax(cont_areas)  # find the largest contour.
        rx, ry, rw, rh = cv2.boundingRect(contours[idx])
        return rx, ry, rw, rh

    def sample_patches(self, img, roi_image, pos_cutoff=.9, neg_cutoff=.1, hard_center=True):
        patch_size = self.patch_size
        rng = np.random.RandomState(self.seed or None)
        roi_f = (np.array(roi_image) > 0).astype(float)
        roi_size = roi_f.sum()
        patch_cover = cv2.blur(roi_f, (patch_size, patch_size))
        roi_cover = (patch_cover * (patch_size ** 2)) / roi_size

        # TODO maybe tad extreme?
        # edge case - might be less then thresh
        image_thresh = max(patch_cover.max(), roi_cover.max())
        pos_cutoff = image_thresh * .95 if image_thresh < pos_cutoff else pos_cutoff

        # abnormalities
        abn_filter = (patch_cover > pos_cutoff) | (roi_cover > pos_cutoff)
        abn_targets = np.argwhere(abn_filter)

        nb_bkg = sum(np.random.binomial(1, self.prob_bkg, self.patch_num))
        nb_abn = self.patch_num - nb_bkg

        # force center in image and sample
        targets = []
        if nb_abn > 0:
            abn_targets = abn_targets[(abn_targets[:, 0] + (patch_size // 2) < roi_f.shape[0]) &
                                      (abn_targets[:, 0] - (patch_size // 2) > 0) &
                                      (abn_targets[:, 1] + (patch_size // 2) < roi_f.shape[1]) &
                                      (abn_targets[:, 1] - (patch_size // 2) > 0)]
            if hard_center:
                upleft_x, upleft_y, rw, rh = self.get_cont(np.array(roi_image))
                abn_targets = abn_targets[(abn_targets[:, 0] < upleft_y + rh) &
                                          (abn_targets[:, 0] > upleft_y) &
                                          (abn_targets[:, 1] < upleft_x + rw) &
                                          (abn_targets[:, 1] > upleft_x)]
            assert len(abn_targets) > 0  # TODO make sure no funny buisness
            targets = abn_targets[rng.choice(len(abn_targets), nb_abn)]

        # background
        if nb_bkg > 0:
            bkg_filter = (patch_cover < neg_cutoff) & (roi_cover < neg_cutoff)
            bkg_targets = np.argwhere(bkg_filter)
            # force center in image and sample
            bkg_targets = bkg_targets[(bkg_targets[:, 0] + (patch_size // 2) < roi_f.shape[0]) &
                                      (bkg_targets[:, 0] - (patch_size // 2) > 0) &
                                      (bkg_targets[:, 1] + (patch_size // 2) < roi_f.shape[1]) &
                                      (bkg_targets[:, 1] - (patch_size // 2) > 0)]
            assert len(bkg_targets) > 0  # TODO make sure no funny buisness
            bkg_targets = bkg_targets[rng.choice(len(bkg_targets), nb_bkg)]
            targets = np.concatenate([targets, bkg_targets]) if targets != [] else bkg_targets

        patches = []
        for target_x, target_y in targets:
            patch = img.crop((target_y - patch_size // 2, target_x - patch_size // 2,
                              target_y + patch_size // 2, target_x + patch_size // 2))
            patches.append(self.patch_transform(patch))
        return patches, nb_abn, nb_bkg

    def __getitem__(self, index):
        # Open image
        patches, nb_abn, nb_bkg = self.sample_patches(self.mam_im[index], self.roi_im[index])
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = ([self.label_arr[index]] * nb_abn) + ([0] * nb_bkg)

        return (patches, single_image_label)

    def __len__(self):
        return self.data_len