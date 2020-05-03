import torch
import warnings
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
from CBIS_dataloader import CBIS_Dataloader, CBIS_PatchDataSet_INMEM
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from sklearn.metrics import roc_curve
import numpy as np
from models import PreBuildConverter
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
import os
import pandas as pd
from models import three_step_params
from conf_table_TB import plot_confusion_matrix
plt.switch_backend('agg')

class PatchLearner(object):
    def __init__(self, conf):

        # -----------   define model --------------- #
        build_model = PreBuildConverter(in_channels=1, out_classes=5, add_soft_max=True, pretrained=conf.pre_train)
        self.models = []
        for _ in range(conf.n_models):
            self.models.append(build_model.get_by_str(conf.net_mode).to(conf.device))
        print('{} {} models generated'.format(conf.n_models, conf.net_mode))

        # ------------  define params -------------- #
        self.milestones = conf.milestones
        self.writer = SummaryWriter(logdir=conf.log_path)
        self.step = 0
        self.epoch = 0
        print('two model heads generated')

        self.get_opt(conf)
        """
        paras_only_bn = []
        paras_wo_bn = []
        for model in self.models:
            paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
            paras_only_bn.append(paras_only_bn_)
            paras_wo_bn.append(paras_wo_bn_)

        self.optimizer = optim.SGD([
                                       {'params': paras_wo_bn[model_num],
                                        'weight_decay': 5e-4}
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr, momentum=conf.momentum)
        """
        print(self.optimizer)

        # ------------  define loaders -------------- #

        self.loader = CBIS_Dataloader(n_patches=2, conf=conf,
                                      og_resize=(1152, 896), patch_size=225, roi_sampling_ratio=.5)

        self.eval_loader = CBIS_Dataloader(n_patches=2, conf=conf,
                                      og_resize=(1152, 896), patch_size=225, roi_sampling_ratio=.5)

        print('optimizers generated')
        self.running_loss = 0.
        self.running_pearson_loss = 0.
        self.running_ensemble_loss = 0.

        self.board_loss_every = max(self.loader.train_len // 10, 1)
        self.evaluate_every = conf.evaluate_every
        self.save_every = max(conf.epoch_per_save, 1)
        assert self.save_every >= self.evaluate_every

    def get_opt(self, conf):
        paras_only_bn = []
        paras_wo_bn = []
        for model in self.models:
            paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
            paras_only_bn.append(paras_only_bn_)
            paras_wo_bn.append(paras_wo_bn_)

        self.optimizer = optim.SGD([
                                       {'params': paras_wo_bn[model_num]}  # , 'weight_decay': 5e-4} #TODO check
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr, momentum=conf.momentum)

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

    def evaluate(self, conf, model_num, mode='test'):
        model = self.models[model_num]
        model.eval()

        n_classes = 5
        predictions = []
        prob = []
        labels = []
        sample_ratio = .25
        loader = self.eval_loader.get_loader(mode, sample=sample_ratio)
        tot = len(self.eval_loader.train_table) if mode=='train' else len(self.eval_loader.test_table)
        tot = tot * sample_ratio // self.eval_loader.n_src_per_batch
        with torch.no_grad():
            for imgs, label in tqdm(loader, total=tot, desc='valid_'+str(model_num), position=1):
                imgs = imgs.to(conf.device)

                self.optimizer.zero_grad()
                theta = model(imgs).detach()

                val, arg = torch.max(theta, dim=1)
                predictions.append(arg.cpu().numpy())
                prob.append(theta.cpu().numpy())
                labels.append(label.detach().cpu().numpy())

        predictions = np.hstack(predictions)
        prob = np.vstack(prob)
        labels = np.hstack(labels)

        # Compute ROC curve and ROC area for each class
        res = (predictions == labels)
        acc = sum(res) / len(res)
        fpr, tpr, _ = roc_curve(np.repeat(res, n_classes), prob.ravel())
        buf = gen_plot(fpr, tpr)
        roc_curve_im = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve_im)
        return acc, roc_curve_tensor

    def pretrain(self, conf):
        for model_num in range(conf.n_models):
            self.models[model_num].train()
            #if not conf.cpu_mode:
            #    self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=[0, 1, 2, 3])
            self.models[model_num].to(conf.device)

        # Stage 1: train only the last dense layer if using pretrained model.
        for model_num in range(conf.n_models):
            for i, (name, param) in enumerate(self.models[model_num].named_parameters()):
                param.requires_grad = (i > three_step_params[conf.net_mode][0])
        self.get_opt(conf)
        self.train(conf, conf.pre_steps[0])

        # Stage 2: train only the top layers.
        for model_num in range(conf.n_models):
            for i, (name, param) in enumerate(self.models[model_num].named_parameters()):
                param.requires_grad = (i > three_step_params[conf.net_mode][1])
        self.train(conf, conf.pre_steps[1])

        # Stage 3: train all layers.
        for model_num in range(conf.n_models):
            for i, (name, param) in enumerate(self.models[model_num].named_parameters()):
                param.requires_grad = True
        self.train(conf, conf.pre_steps[2])

    def train(self, conf, epochs):
        if not conf.pre_train:
            for model_num in range(conf.n_models):
                self.models[model_num].train()
                #if not conf.cpu_mode:
                #    self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=[0, 1, 2, 3])
                self.models[model_num].to(conf.device)

            self.running_loss = 0.
            self.running_pearson_loss = 0.
            self.running_ensemble_loss = 0.
        epoch_iter = range(epochs)
        for e in epoch_iter:
            # check lr update
            for milestone in self.milestones:
                if self.epoch == milestone:
                    self.schedule_lr()

            loader = self.loader.get_loader('train')
            # for imgs, labels in self.loader.get_loader('tarin'):
            for imgs, labels in tqdm(loader, desc='epoch {}'.format(e), total=self.loader.train_len, position=0):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                joint_losses = []
                for model_num in range(conf.n_models):
                    theta = self.models[model_num](imgs)
                    thetas.append(theta)
                    joint_losses.append(conf.ce_loss(theta, labels))
                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)

                # calc loss
                if conf.pearson:
                    outputs = torch.stack(thetas)
                    pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    self.running_pearson_loss += pearson_corr_models_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    self.running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses * 0.5 + alpha * ensemble_loss
                else:
                    loss = joint_losses

                loss.backward()
                self.running_loss += loss.item()
                self.optimizer.step()

                # listen to running losses
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = self.running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    self.running_loss = 0.

                    if conf.pearson:  # ganovich listening to pearson
                        loss_board = self.running_pearson_loss / self.board_loss_every
                        self.writer.add_scalar('pearson_loss', loss_board, self.step)
                        self.running_pearson_loss = 0.

                    if conf.joint_mean:
                        loss_board = self.running_ensemble_loss / self.board_loss_every
                        self.writer.add_scalar('ensemble_loss', loss_board, self.step)
                        self.running_ensemble_loss = 0.

                # listen to validation and save every so often
                if self.step % (self.loader.train_len//2) == 0 and self.step != 0:
                    for model_num in range(conf.n_models):
                        accuracy, roc_curve_tensor = self.evaluate(conf=conf, model_num=model_num, mode='test')
                        self.board_val('mod_test_' + str(model_num), accuracy, roc_curve_tensor)
                        self.models[model_num].train()
                if self.step % (self.loader.train_len) == 0 and self.step != 0:
                    for model_num in range(conf.n_models):
                        accuracy, roc_curve_tensor = self.evaluate(conf=conf, model_num=model_num, mode='train')
                        self.board_val('mod_train_' + str(model_num), accuracy, roc_curve_tensor)
                        self.models[model_num].train()
                self.step += 1

            if self.epoch % self.save_every == 0 and self.epoch != 0:
                self.save_state(conf, accuracy)
            self.epoch += 1

        if accuracy is not None:
            self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)


class PatchLearnerMult(object):
    def __init__(self, conf):

        # -----------   define model --------------- #
        n_classes = (2 if (conf.type_only or conf.cancer_only) else 4) + (0 if conf.no_bkg else 1)
        build_model = PreBuildConverter(in_channels=1, out_classes=n_classes, add_soft_max=True,
                                        pretrained=conf.pre_train, half=conf.half)
        self.models = []
        for _ in range(conf.n_models):
            self.models.append(build_model.get_by_str(conf.net_mode).to(conf.device))
        print('{} {} models generated'.format(conf.n_models, conf.net_mode))

        # ------------  define params -------------- #
        self.milestones = conf.milestones
        if not os.path.exists(conf.log_path):
            os.mkdir(conf.log_path)
        self.writer = SummaryWriter(logdir=conf.log_path)
        self.step = 0
        self.epoch = 0
        print('two model heads generated')

        self.get_opt(conf)
        print(self.optimizer)

        # ------------  define loaders -------------- #

        self.train_ds = CBIS_PatchDataSet_INMEM(mode='train', patch_num=conf.n_patch, prob_bkg=conf.bkg_prob,
                                                no_bkg=conf.no_bkg, cancer_only=conf.cancer_only, type_only=conf.type_only)
        self.test_ds = CBIS_PatchDataSet_INMEM(mode='test', patch_num=conf.n_patch, prob_bkg=conf.bkg_prob,
                                               no_bkg=conf.no_bkg, cancer_only=conf.cancer_only, type_only=conf.type_only)

        dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }

        train_weights = pd.Series(self.train_ds.label_arr).value_counts()
        train_weights = (len(self.train_ds.label_arr) / pd.Series(self.train_ds.label_arr).map(train_weights)).values
        test_weights = pd.Series(self.test_ds.label_arr).value_counts()
        test_weights = (len(self.test_ds.label_arr) / pd.Series(self.test_ds.label_arr).map(test_weights)).values

        self.train_loader = DataLoader(self.train_ds,
                                       sampler=WeightedRandomSampler(train_weights, len(train_weights)), **dloader_args)
        self.test_loader = DataLoader(self.test_ds,
                                      sampler=WeightedRandomSampler(test_weights, len(test_weights)), **dloader_args)

        self.eval_train = DataLoader(self.train_ds,
                                       sampler=WeightedRandomSampler(train_weights, len(train_weights) // 10), **dloader_args)
        self.eval_test = DataLoader(self.test_ds,
                                      sampler=WeightedRandomSampler(test_weights, len(test_weights) // 2), **dloader_args)

        """
        #eval_train_sampler = RandomSampler(self.train_ds, replacement=True, num_samples=len(self.train_ds) // 10)
        #eval_test_sampler = RandomSampler(self.test_ds, replacement=True, num_samples=len(self.test_ds) // 2)
        #self.eval_train = DataLoader(self.train_ds, sampler=eval_train_sampler, **dloader_args)
        #self.eval_test = DataLoader(self.test_ds, sampler=eval_test_sampler, **dloader_args)
        """

        print('optimizers generated')
        self.running_loss = 0.
        self.running_pearson_loss = 0.
        self.running_ensemble_loss = 0.

        self.board_loss_every = max(len(self.train_loader) // 4, 1)
        self.evaluate_every = conf.evaluate_every
        self.save_every = max(conf.epoch_per_save, 1)
        assert self.save_every >= self.evaluate_every

    def get_opt(self, conf):
        paras_only_bn = []
        paras_wo_bn = []
        for model in self.models:
            paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
            paras_only_bn.append(paras_only_bn_)
            paras_wo_bn.append(paras_wo_bn_)

        self.optimizer = optim.SGD([
                                       {'params': paras_wo_bn[model_num], 'weight_decay': 5e-4}
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr, momentum=conf.momentum)

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

    def board_val(self, db_name, accuracy, roc_curve_tensor, cm_fig):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        self.writer.add_figure('{}_conf_table'.format(db_name), cm_fig, self.step)

    def evaluate_alt(self, conf, mode='test'):
        for i in range(len(self.models)):
            self.models[i].eval()
        # TODO look into this https://github.com/pytorch/pytorch/issues/11476
        # batching is unstable... limit to less gpus or use sync
        n_classes = (2 if (conf.type_only or conf.cancer_only) else 4) + (0 if conf.no_bkg else 1)
        predictions = dict.fromkeys(range(-1, len(self.models)), [])
        prob = dict.fromkeys(range(-1, len(self.models)), [])
        labels = []
        loader = self.eval_train if mode == 'train' else self.eval_test
        pos = 2 if mode == 'train' else 1
        with torch.no_grad():
            for imgs, label in tqdm(loader, total=len(loader), desc=mode, position=pos):
                imgs = torch.cat(imgs).to(conf.device)

                self.optimizer.zero_grad()
                thetas = [model(imgs).detach() for model in self.models]
                thetas = [torch.mean(torch.stack(thetas), 0)] + thetas
                for ind, theta in zip(range(-1, len(self.models)), thetas):
                    val, arg = torch.max(theta, dim=1)
                    predictions[ind].append(arg.cpu().numpy())
                    prob[ind].append(theta.cpu().numpy())
                labels.append(torch.cat(label).detach().cpu().numpy())

        labels = np.hstack(labels)
        results = []
        for ind in range(-1, len(self.models)):
            predictions[ind] = np.hstack(predictions[ind])
            prob[ind] = np.vstack(prob[ind])

            # Compute ROC curve and ROC area for each class
            res = (predictions[ind] == labels)
            acc = sum(res) / len(res)
            fpr, tpr, _ = roc_curve(np.repeat(res, n_classes), prob[ind].ravel())
            buf = gen_plot(fpr, tpr)
            roc_curve_im = Image.open(buf)
            roc_curve_tensor = trans.ToTensor()(roc_curve_im)
            results.append((acc, roc_curve_tensor))
        return results

    def evaluate(self, conf, model_num, mode='test'):
        if model_num == -1:  # means mean model
            for i in range(len(self.models)):
                self.models[i].eval()
        else:
            model = self.models[model_num]
            model.eval()
        # TODO look into this https://github.com/pytorch/pytorch/issues/11476
        # batching is unstable... limit to less gpus or use sync
        n_classes = (2 if (conf.type_only or conf.cancer_only) else 4) + (0 if conf.no_bkg else 1)
        label_names = []
        if conf.type_only:
            label_names = ['calc', 'mass']
        elif conf.cancer_only:
            label_names = ['mal', 'ben']
        else:
            label_names = ['calc_mal', 'calc_ben', 'mass_mal', 'mass_ben']

        if not conf.no_bkg:
            label_names = ['bkg'] + label_names

        predictions = []
        prob = []
        labels = []
        loader = self.eval_train if mode == 'train' else self.eval_test
        pos = 2 if mode == 'train' else 1
        model_num_str = model_num if model_num > -1 else 'mean'
        with torch.no_grad():
            for imgs, label in tqdm(loader, total=len(loader), desc=mode+'_'+str(model_num_str), position=pos):
                imgs = torch.cat(imgs).to(conf.device)

                self.optimizer.zero_grad()
                if model_num == -1: # means mean model
                    theta = torch.mean(torch.stack([model(imgs).detach() for model in self.models]), 0)
                else:
                    theta = model(imgs).detach()

                val, arg = torch.max(theta, dim=1)
                predictions.append(arg.cpu().numpy())
                prob.append(theta.cpu().numpy())
                labels.append(torch.cat(label).detach().cpu().numpy())

        predictions = np.hstack(predictions)
        prob = np.vstack(prob)
        labels = np.hstack(labels)

        # Compute ROC curve and ROC area for each class
        img_d_fig = plot_confusion_matrix(labels, predictions, label_names, tensor_name='dev/cm_' + mode)
        res = (predictions == labels)
        acc = sum(res) / len(res)
        fpr, tpr, _ = roc_curve(np.repeat(res, n_classes), prob.ravel())
        buf = gen_plot(fpr, tpr)
        roc_curve_im = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve_im)
        return acc, roc_curve_tensor, img_d_fig

    def pretrain(self, conf):
        for model_num in range(conf.n_models):
            self.models[model_num].train()
            if not conf.cpu_mode:
                device_ids = list(range(conf.ngpu))
                self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=device_ids)
            self.models[model_num].to(conf.device)

        pre_layers = three_step_params[conf.net_mode] if len(conf.pre_layers) < 1 else conf.pre_layers
        assert len(pre_layers) == len(conf.pre_steps)
        for layer_step, layer_epoch in zip(pre_layers, conf.pre_steps):
            for model_num in range(conf.n_models):
                for i, (name, param) in enumerate(self.models[model_num].named_parameters()):
                    param.requires_grad = (i > layer_step) or ('bn' in name)
            self.train(conf, layer_epoch)

    def train(self, conf, epochs):
        if not conf.pre_train:
            for model_num in range(conf.n_models):
                self.models[model_num].train()
                if not conf.cpu_mode:
                    device_ids = list(range(conf.ngpu))
                    self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=device_ids)
                self.models[model_num].to(conf.device)

            self.running_loss = 0.
            self.running_pearson_loss = 0.
            self.running_ensemble_loss = 0.
        epoch_iter = range(epochs)
        accuracy = 0
        for e in epoch_iter:
            # check lr update
            for milestone in self.milestones:
                if self.epoch == milestone:
                    self.schedule_lr()

            # train
            for imgs, labels in tqdm(self.train_loader, desc='epoch {}'.format(e), total=len(self.train_loader), position=0):
                imgs = torch.cat(imgs).to(conf.device, dtype=torch.half if conf.half else None)
                labels = torch.cat(labels).to(conf.device)

                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                joint_losses = []
                for model_num in range(conf.n_models):
                    theta = self.models[model_num](imgs)
                    thetas.append(theta)
                    joint_losses.append(conf.ce_loss(theta, labels))
                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)

                if self.step == 70:
                    pass

                # calc loss
                if conf.pearson:
                    outputs = torch.stack(thetas)
                    pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    self.running_pearson_loss += pearson_corr_models_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    self.running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses * 0.5 + alpha * ensemble_loss
                else:
                    loss = joint_losses

                loss.backward()
                self.running_loss += loss.item()
                self.optimizer.step()

                # listen to running losses
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = self.running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    self.running_loss = 0.
                    if conf.pearson:  # ganovich listening to pearson
                        loss_board = self.running_pearson_loss / self.board_loss_every
                        self.writer.add_scalar('pearson_loss', loss_board, self.step)
                        self.running_pearson_loss = 0.

                    if conf.joint_mean:
                        loss_board = self.running_ensemble_loss / self.board_loss_every
                        self.writer.add_scalar('ensemble_loss', loss_board, self.step)
                        self.running_ensemble_loss = 0.

                self.step += 1

            # TODO replace 3 pass with 1 pass
            # listen to validation and save every so often
            if self.epoch % self.evaluate_every == 0 and self.epoch != 0:
                for mode in ['test', 'train']:
                    # results = self.evaluate_alt(conf=conf, mode='test')
                    # for model_num, (accuracy, roc_curve_tensor) in zip(range(-1, conf.n_models), results):
                    for model_num in range(-1 if conf.n_models > 1 else 0, conf.n_models):
                        accuracy, roc_curve_tensor, img_d_fig = self.evaluate(conf=conf, model_num=model_num, mode=mode)
                        broad_name = 'mod_'+mode+'_' + str(model_num if model_num>-1 else 'mean')
                        self.board_val(broad_name, accuracy, roc_curve_tensor, img_d_fig)
                        if model_num > -1: self.models[model_num].train()

            if self.epoch % self.save_every == 0 and self.epoch != 0:
                self.save_state(conf, accuracy)
            self.epoch += 1

        if accuracy is not None:
            self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)


class PatchLearnerMultDist(object):
    def __init__(self, conf):

        # -----------   define model --------------- #
        build_model = PreBuildConverter(in_channels=1, out_classes=5, add_soft_max=True)
        self.models = []
        for _ in range(conf.n_models):
            self.models.append(build_model.get_by_str(conf.net_mode).to(conf.device))
        print('{} {} models generated'.format(conf.n_models, conf.net_mode))

        # ------------  define params -------------- #
        self.milestones = conf.milestones
        if not os.path.exists(conf.log_path):
            os.mkdir(conf.log_path)
        os.mkdir(conf.log_path / str(conf.local_rank))
        self.writer = SummaryWriter(logdir=conf.log_path / str(conf.local_rank))
        self.step = 0
        self.epoch = 0
        print('two model heads generated')

        paras_only_bn = []
        paras_wo_bn = []
        for model in self.models:
            paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
            paras_only_bn.append(paras_only_bn_)
            paras_wo_bn.append(paras_wo_bn_)

        self.optimizer = optim.SGD([
                                       {'params': paras_wo_bn[model_num],
                                        'weight_decay': 5e-4}
                                       for model_num in range(conf.n_models)
                                   ] + [
                                       {'params': paras_only_bn[model_num]}
                                       for model_num in range(conf.n_models)
                                   ], lr=conf.lr, momentum=conf.momentum)
        print(self.optimizer)

        # ------------  define loaders -------------- #

        self.train_ds = CBIS_PatchDataSet_INMEM(mode='train', nb_abn=conf.n_patch, nb_bkg=conf.n_patch)
        self.test_ds = CBIS_PatchDataSet_INMEM(mode='test', nb_abn=conf.n_patch, nb_bkg=conf.n_patch)

        self.train_sampler = DistributedSampler(self.train_ds, num_replicas=4, rank=conf.local_rank)
        self.test_sampler = DistributedSampler(self.train_ds, num_replicas=4, rank=conf.local_rank)

        dloader_args = {
            'batch_size': conf.batch_size,
            'pin_memory': True,
            'num_workers': conf.num_workers,
            'drop_last': False,
        }

        self.train_loader = DataLoader(self.train_ds, sampler=self.train_sampler, **dloader_args)
        self.test_loader = DataLoader(self.test_ds, sampler=self.test_sampler, **dloader_args)

        eval_train_sampler = RandomSampler(self.train_ds, replacement=True, num_samples=len(self.train_ds) // 10)
        eval_test_sampler = RandomSampler(self.test_ds, replacement=True, num_samples=len(self.test_ds) // 2)
        self.eval_train = DataLoader(self.train_ds, sampler=eval_train_sampler, **dloader_args)
        self.eval_test = DataLoader(self.test_ds, sampler=eval_test_sampler, **dloader_args)

        print('optimizers generated')
        self.board_loss_every = max(len(self.train_loader) // 4, 1)
        self.evaluate_every = conf.evaluate_every
        self.save_every = max(conf.epoch_per_save, 1)
        assert self.save_every >= self.evaluate_every

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

    def evaluate(self, conf, model_num, mode='test'):
        model = self.models[model_num]
        model.eval()
        # TODO look into this https://github.com/pytorch/pytorch/issues/11476
        # batching is unstable... limit to less gpus or use sync
        n_classes = 5
        predictions = []
        prob = []
        labels = []
        confidance = []
        loader = self.eval_train if mode == 'train' else self.eval_test
        pos = 2 if mode == 'train' else 1
        with torch.no_grad():
            for imgs, label in tqdm(loader, total=len(loader), desc=mode+'_'+str(model_num), position=pos):
                imgs = torch.cat(imgs).to(conf.device)

                self.optimizer.zero_grad()
                theta = model(imgs).detach()
                val, arg = torch.max(theta, dim=1)
                confidance.append(val.cpu().numpy())
                predictions.append(arg.cpu().numpy())
                prob.append(theta.cpu().numpy())
                labels.append(torch.cat(label).detach().cpu().numpy())

        predictions = np.hstack(predictions)
        confidance = np.hstack(confidance)
        prob = np.vstack(prob)
        labels = np.hstack(labels)

        # Compute ROC curve and ROC area for each class
        res = (predictions == labels)
        acc = sum(res) / len(res)
        fpr, tpr, _ = roc_curve(res, confidance)
        buf = gen_plot(fpr, tpr)
        roc_curve_im = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve_im)
        return acc, roc_curve_tensor

    def train(self, conf, epochs):
        torch.cuda.set_device(conf.local_rank)
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=4,
            rank=conf.local_rank,
        )
        for model_num in range(conf.n_models):
            self.models[model_num].train()
            if not conf.cpu_mode:
                #self.models[model_num] = DataParallel(self.models[model_num], device_ids=[0, 1, 2, 3])
                self.models[model_num] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models[model_num])
                self.models[model_num] = torch.nn.parallel.DistributedDataParallel(self.models[model_num],
                                                        device_ids=[conf.local_rank], output_device=conf.local_rank)
            self.models[model_num].to(conf.device)

        self.running_loss = 0.
        self.running_pearson_loss = 0.
        self.running_ensemble_loss = 0.
        epoch_iter = range(epochs)
        for e in epoch_iter:
            # check lr update
            for milestone in self.milestones:
                if self.epoch == milestone:
                    self.schedule_lr()

            # train
            for imgs, labels in tqdm(self.train_loader, desc='epoch {}'.format(e), total=len(self.train_loader), position=0):
                imgs = torch.cat(imgs).to(conf.device)
                labels = torch.cat(labels).to(conf.device)

                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                joint_losses = []
                for model_num in range(conf.n_models):
                    theta = self.models[model_num](imgs)
                    thetas.append(theta)
                    joint_losses.append(conf.ce_loss(theta, labels))
                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)

                # calc loss
                if conf.pearson:
                    outputs = torch.stack(thetas)
                    pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    self.running_pearson_loss += pearson_corr_models_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    self.running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses * 0.5 + alpha * ensemble_loss
                else:
                    loss = joint_losses

                loss.backward()
                self.running_loss += loss.item()
                self.optimizer.step()

                # listen to running losses
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = self.running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    self.running_loss = 0.

                    if conf.pearson:  # ganovich listening to pearson
                        loss_board = self.running_pearson_loss / self.board_loss_every
                        self.writer.add_scalar('pearson_loss', loss_board, self.step)
                        self.running_pearson_loss = 0.

                    if conf.joint_mean:
                        loss_board = self.running_ensemble_loss / self.board_loss_every
                        self.writer.add_scalar('ensemble_loss', loss_board, self.step)
                        self.running_ensemble_loss = 0.

                self.step += 1

            # listen to validation and save every so often
            if self.epoch % self.evaluate_every == 0 and self.epoch != 0:
                for model_num in range(conf.n_models):
                    accuracy, roc_curve_tensor = self.evaluate(conf=conf, model_num=model_num, mode='test')
                    self.board_val('mod_test_' + str(model_num), accuracy, roc_curve_tensor)
                    self.models[model_num].train()
            if self.epoch % self.evaluate_every == 0 and self.epoch != 0:
                for model_num in range(conf.n_models):
                    accuracy, roc_curve_tensor = self.evaluate(conf=conf, model_num=model_num, mode='train')
                    self.board_val('mod_train_' + str(model_num), accuracy, roc_curve_tensor)
                    self.models[model_num].train()

            if self.epoch % self.save_every == 0 and self.epoch != 0:
                self.save_state(conf, accuracy)
            self.epoch += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)
