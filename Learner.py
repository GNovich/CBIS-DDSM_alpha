import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
from torchvision.models import resnet50
from torch.nn import Conv2d, Linear
from functools import partial
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_curve
import numpy as np
import cv2
plt.switch_backend('agg')


class Learner(object):
    def __init__(self, conf, inference=False):
        print(conf)

        # -----------   define model --------------- #
        def base_model():
            # a very basic resnet50 for greyscale images with binary res
            model = resnet50(pretrained=False)
            model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = Linear(in_features=2048, out_features=2, bias=True)
            return model

        self.models = [base_model().to(conf.device) for _ in range(conf.n_models)]
        print('{}_{} models generated'.format(conf.net_mode, conf.net_depth))

        # ------------  define loaders -------------- #
        dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False, # check that it fits in mem
            'shuffle': True
        }

        grey_loader = partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
        file_ext = ('.png',)
        im_trans = conf.im_transform
        self.dataset = DatasetFolder(conf.train_folder, extensions=file_ext, loader=grey_loader, transform=im_trans)
        valid_size = int(np.floor(conf.valid_ratio * len(self.dataset)))
        train_size = len(self.dataset) - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        self.train_loader = DataLoader(train_dataset, **dloader_args)
        self.valid_loader = DataLoader(valid_dataset, **dloader_args)

        self.test_ds = DatasetFolder(conf.test_folder, extensions=file_ext, loader=grey_loader, transform=im_trans)
        self.test_loader = DataLoader(self.test_ds, **dloader_args)

        if conf.morph_dir:
            self.morph_ds = DatasetFolder(conf.morph_dir, extensions=file_ext, loader=grey_loader, transform=im_trans)
            self.morph_loader = DataLoader(self.morph_ds, **dloader_args)
        else:
            self.morph_loader = []

        # ------------  define params -------------- #
        self.inference = inference
        if not inference:
            self.milestones = conf.milestones
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            print('two model heads generated')

            paras_only_bn = []
            paras_wo_bn = []
            for model in self.models:
                paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
                paras_only_bn.append(paras_only_bn_)
                paras_wo_bn.append(paras_wo_bn_)

            self.optimizer = optim.SGD([
                                           {'params': paras_only_bn[model_num]}
                                           for model_num in range(conf.n_models)
                                       ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)

            print('optimizers generated')
            self.board_loss_every = max(len(self.train_loader) // 100, 1)
            self.evaluate_every = max(len(self.train_loader) // 10, 1)
            self.save_every = max(len(self.train_loader) // conf.save_per_epoch, 1)

        else:
            self.threshold = conf.threshold

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        for mod_num in range(conf.n_models):
            torch.save(
                self.models[mod_num].state_dict(), save_path /
                                                   ('model_{}_{}_accuracy:{}_step:{}_{}.pth'.format(mod_num, get_time(),
                                                                                                    accuracy, self.step,
                                                                                                    extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        def load_fix(target_path):
            a = torch.load(target_path)
            fixed_a = {k.split('module.')[-1]: a[k] for k in a}
            torch.save(fixed_a, target_path)

        for mod_num in range(conf.n_models):
            target_path = save_path / 'model_{}_{}'.format(mod_num, fixed_str)
            load_fix(target_path)
            self.models[mod_num].load_state_dict(torch.load(target_path))
        if not model_only:
            for mod_num in range(conf.n_models):
                target_path = save_path / 'head_{}_{}'.format(mod_num, fixed_str)
                load_fix(target_path)
                self.heads[mod_num].load_state_dict(torch.load(target_path))
            target_path = save_path / 'optimizer_{}'.format(fixed_str)
            load_fix(target_path)
            self.optimizer.load_state_dict(torch.load(target_path))

    def board_val(self, db_name, accuracy, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    def evaluate(self, conf, model_num):
        model = self.models[model_num]
        model.eval()

        predictions = []
        prob = []
        labels = []
        with torch.no_grad():
            for imgs, label in tqdm(self.valid_loader, total=len(self.valid_loader), desc='valid', position=0):
                imgs = imgs.to(conf.device)

                self.optimizer.zero_grad()
                theta = model(imgs).detach()

                val, arg = torch.max(theta, dim=1)
                predictions.append(arg.cpu().numpy())
                prob.append(val.cpu().numpy())
                labels.append(label.detach().cpu().numpy())
        predictions = np.hstack(predictions)
        prob = np.hstack(prob)
        labels = np.hstack(labels)

        res = (predictions == labels)
        acc = sum(res) / len(res)
        tpr, fpr, _ = roc_curve(res, prob)
        buf = gen_plot(fpr, tpr)
        roc_curve_im = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve_im)
        return acc, roc_curve_tensor

    def train(self, conf, epochs):
        for model_num in range(conf.n_models):
            self.models[model_num].train()
            if not conf.cpu_mode:
                self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=[0, 1, 2, 3])
            self.models[model_num].to(conf.device)

        running_loss = 0.
        running_pearson_loss = 0.
        running_ensemble_loss = 0.
        running_morph_loss = 0.
        morph_iter = iter(self.morph_loader)
        morph_load = bool(conf.morph_dir)
        epoch_iter = range(epochs)
        for e in epoch_iter:
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(self.train_loader, desc='epoch {}'.format(e), position=1):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                # moprh_inf_loop
                if morph_load:
                    try:
                        morphs, morph_labels = next(morph_iter)
                    except StopIteration:
                        morph_iter = iter(self.morph_loader)
                        morphs, morph_labels = next(morph_iter)

                use_morph = morph_load and (morphs[0, 0, 0, 0] < 100).item()
                # this is an inheritance hack in the loader TODO replace with a better one
                if use_morph:
                    morphs = morphs.to(conf.device)
                    morph_labels = morph_labels.to(conf.device)

                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                joint_losses = []
                morph_thetas = []
                for model in self.models:
                    if use_morph:
                        cat_inputs = torch.cat([imgs, morphs])
                        cat_emb = model(cat_inputs)
                        theta = cat_emb[:imgs.shape[0], :]
                        theta_morph = cat_emb[imgs.shape[0]:, :]
                        thetas.append(theta)
                        morph_thetas.append(theta_morph)
                        joint_losses.append(conf.ce_loss(theta, labels))
                    else:
                        theta = model(imgs)
                        thetas.append(theta)
                        joint_losses.append(conf.ce_loss(theta, labels))

                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)

                # calc loss
                if conf.pearson:
                    outputs = torch.stack(thetas)
                    pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    running_pearson_loss += pearson_corr_models_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses * 0.5 + alpha * ensemble_loss
                else:
                    loss = joint_losses

                # Morph loss
                # Make sure models do not produce the same results as the morphs
                if use_morph:
                    morph_loss = []
                    for morph_theta in morph_thetas:
                        mask = torch.nn.functional.one_hot(morph_labels, num_classes=morph_theta.shape[-1]).type(torch.bool)
                        correct_values = torch.masked_select(morph_theta, mask)
                        average_values = morph_theta.mean(1)
                        morph_loss.append(conf.morph_loss(correct_values, average_values))
                    morph_loss = sum(morph_loss) / max(len(morph_thetas), 1)
                    morph_loss *= conf.morph_alpha
                    running_morph_loss += morph_loss.item()
                    loss = loss + morph_loss

                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                # listen to running losses
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                    if conf.pearson:  # ganovich listening to pearson
                        loss_board = running_pearson_loss / self.board_loss_every
                        self.writer.add_scalar('pearson_loss', loss_board, self.step)
                        running_pearson_loss = 0.

                    if conf.joint_mean:
                        loss_board = running_ensemble_loss / self.board_loss_every
                        self.writer.add_scalar('ensemble_loss', loss_board, self.step)
                        running_ensemble_loss = 0.

                    if conf.morph_dir:
                        loss_board = running_morph_loss / self.board_loss_every
                        self.writer.add_scalar('morph_loss', loss_board, self.step)
                        running_morph_loss = 0.

                # listen to validation and save every so often
                for model_num in range(conf.n_models):
                    # see validation set
                    if self.step % self.evaluate_every == 0 and self.step != 0:
                        accuracy, roc_curve_tensor = self.evaluate(conf=conf, model_num=model_num)
                        self.board_val('mod_val', accuracy, roc_curve_tensor)
                        self.models[model_num].train()
                    if self.step % self.save_every == 0 and self.step != 0:
                        self.save_state(conf, accuracy)

                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)
