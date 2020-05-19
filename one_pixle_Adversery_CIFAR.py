from config import get_config
import argparse
from ShapeLearner import ShapeLearner
from ShapeLoader import ShapeDataSet
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import pickle
import torch
import os
import pathlib
import sys
from itertools import product
import re
from os import path
from tqdm import tqdm
import pandas as pd
import eagerpy as ep
from functools import partial
from foolbox import PyTorchModel
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
sys.path.append('one-pixel-attack-pytorch')
from attack import attack_all as OnePixleAttack

sys.path.append('/mnt/md0/orville/Miriam/modular-loss-experiments-morph/')
from src.models import DenseModel, ConvModel, DenseNet
from src.argument_parser import parse_args
from src.dataloader import get_dataloaders
from src.distributions import distributions
class_dir = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
             5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# Momentum Iterative Method (MIM)
# Jacobian-based Saliency Map Attack (JSMA)

datasets_dict = {'CIFAR-10': {
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
    'clean': transforms.Compose([
        transforms.ToTensor()
    ]),
    'dset_kwargs': {},
    'val_size': 10000,
    'distribution': 'categorical',
    'input_shape': (3, 32, 32),
    'output_dim': 10
}
}


def get_dataloaders_(batch_size, trial_i, dataset='MNIST', augment=False, early_stop=False, use_morph=False, depth=None,
                     n_workers=0):
    data_dir = './data/{}'.format(dataset)

    params = datasets_dict[dataset]

    datasets = {}
    for split in ['train_clean', 'train', 'valid', 'test']:
        if augment and split == 'train' and 'train_transform' in params.keys():
            transform = params['train_transform']
        else:
            transform = params['transform']
        if split == 'train_clean':
            transform = params['clean']

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
        'shuffle': False
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
                                          **default_dloader_args)
        dataloaders['train_clean'] = DataLoader(dataset=datasets['train_clean'],
                                                **default_dloader_args)
        dataloaders['valid'] = DataLoader(dataset=datasets['test'],
                                          **default_dloader_args)
        dataloaders['test'] = DataLoader(dataset=datasets['test'],
                                         **default_dloader_args)

    return dataloaders, params



def prep_learner():
    conf = get_config(2)
    conf.device = 'cpu'
    conf.net_mode = 'resnet18'
    conf.n_shapes = 1
    conf.n_colors = 3
    conf.shape_only = False
    conf.color_only = False
    return ShapeLearner(conf, inference=True)


def set_distractors(learner):
    # set OOD data
    triangle_ds = ShapeDataSet(no_bkg=True)
    triangle_ds.shapes = ['triangle']
    triangle_ds.colors = [[(255, 255), (0, 0), (0, 0)],
                          [(0, 0), (255, 255), (0, 0)],
                          [(0, 0), (0, 0), (255, 255)]]
    triangle_ds.n_shapes = 1
    triangle_ds.n_colors = 3
    ziped_classes = enumerate(product(range(1), range(3)))
    triangle_ds.label_map = {v: -1 for k, v in ziped_classes}
    triangle_ds.label_names = [-1]
    learner.ds = triangle_ds

    dloader_args = {
        'batch_size': 32,
        'pin_memory': True,
        'num_workers': conf.num_workers,
        'drop_last': False,
    }
    learner.loader = DataLoader(learner.ds, **dloader_args)
    eval_sampler = RandomSampler(learner.ds, replacement=True, num_samples=len(learner.ds) // 10)
    learner.eval_loader = DataLoader(learner.ds, sampler=eval_sampler, **dloader_args)
    return learner

def set_probes(learner):
    # set OOD data
    triangle_ds = ShapeDataSet(no_bkg=True)
    triangle_ds.shapes = ['rectangle', 'circle']
    triangle_ds.colors = [[(255, 255), (0, 0), (0, 0)],
                          [(0, 0), (255, 255), (0, 0)],
                          ]  # [(0, 0), (0, 0), (255, 255)]]
    triangle_ds.n_shapes = 2
    triangle_ds.n_colors = 2  # TODO fix this! we need the right 2 in the right order!
    ziped_classes = enumerate(product(range(triangle_ds.n_shapes), range(triangle_ds.n_colors)))
    triangle_ds.label_map = {v: k for k, v in ziped_classes}
    triangle_ds.label_names = [str(x) for x in product(triangle_ds.shapes, range(triangle_ds.n_colors))]
    learner.ds = triangle_ds

    dloader_args = {
        'batch_size': 32,
        'pin_memory': True,
        'num_workers': conf.num_workers,
        'drop_last': False,
    }
    learner.loader = DataLoader(learner.ds, **dloader_args)
    eval_sampler = RandomSampler(learner.ds, replacement=True, num_samples=len(learner.ds) // 10)
    learner.eval_loader = DataLoader(learner.ds, sampler=eval_sampler, **dloader_args)
    return learner

def get_evaluation(learner):
    # evaluate OOD data
    for i in range(len(learner.models)):
        learner.models[i].eval()
    do_mean = -1 if len(learner.models) > 1 else 0
    ind_iter = range(do_mean, len(learner.models))
    predictions = dict(zip(ind_iter, [[] for i in ind_iter]))
    prob = dict(zip(ind_iter, [[] for i in ind_iter]))
    labels = []
    learner.eval_loader.dataset.set_mode('test')  # todo check this works :)
    for imgs, label in tqdm(learner.eval_loader, total=len(learner.eval_loader)):
        imgs = imgs.to(conf.device)
        thetas = [model(imgs).detach() for model in learner.models]
        if len(learner.models) > 1: thetas = [torch.mean(torch.stack(thetas), 0)] + thetas
        for ind, theta in zip(range(do_mean, len(learner.models)), thetas):
            val, arg = torch.max(theta, dim=1)
            predictions[ind].append(arg.cpu().numpy())
            prob[ind].append(theta.cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    labels = np.hstack(labels)
    for ind in range(do_mean, len(learner.models)):
        predictions[ind] = np.hstack(predictions[ind])
        prob[ind] = np.vstack(prob[ind])
    return prob, predictions, labels


class ModelMeanEP(torch.nn.Module):
    def __init__(self, model, device='cpu'):
        super(ModelMeanEP, self).__init__()
        self.model = torch.nn.DataParallel(model, dim=0)

    def forward(self, x):
        res = torch.nn.Softmax(-1)(self.model(x))
        return torch.cat([x.mean(0) for x in torch.chunk(res, 4)])
        #return torch.cat([x.max(0).values for x in torch.chunk(res, 4)])


def run_attacks(res_path):
    MORPH_MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results_morph_correct/CIFAR-10/densenet-82-8-8'
    MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results/CIFAR-10/densenet-82-8-8'
    # UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_2_1581641733617'
    # CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_2_1581641746832'
    # CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_2_1581641777871'
    UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_3_1585505819121'
    CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_3_1585505685528'
    CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_3_1585505042819'

    rel_dirs = [UNCORR_MODEL_DIR, CORR_MODEL_DIR, CORR_MODEL_DIR_2]
    alpha = ['0', '0.1', '0.2']

    res = dict.fromkeys(alpha)
    batch_size = 516
    n_workers = 20
    dataset = 'CIFAR-10'
    network = 'densenet-82-8-8'
    loaders, _ = get_dataloaders_(batch_size, 0, dataset, False, early_stop=False, n_workers=n_workers)
    n_models = 3

    params = {}
    params['densenet-82-8-8'] = {'num_modules': 2, 'bottleneck': True, 'reduction': 0.5, 'depth': 82, 'growth_rate': 8,
                                 'input_shape': (3, 32, 32), 'output_dim': 10}
    network = 'densenet-82-8-8'
    model = DenseNet(input_shape=params[network]['input_shape'],
                     output_dim=params[network]['output_dim'],
                     growth_rate=params[network]['growth_rate'],
                     depth=params[network]['depth'],
                     reduction=params[network]['reduction'],
                     bottleneck=params[network]['bottleneck'],
                     num_modules=n_models)

    device = torch.device("cuda")

    for model_path, curr_alpha in tqdm(zip(rel_dirs, alpha), total=len(alpha)):
        weight_path = path.join(MODEL_DIR, model_path, 'trial_0/0.0/weights/final_weights.pt')
        model.reset_parameters()
        model.load_state_dict(torch.load(weight_path))
        model.eval()  # model.train(mode=False)
        fmodel = PyTorchModel(ModelMeanEP(model), bounds=(0, 1), device=device)

        res[curr_alpha] = dict()
        for (attack, eps), attack_name in tqdm(zip(attack_list, attack_list_names),
                                                        desc='attaking ' + str(curr_alpha), total=len(attack_list)):
            attack = attack()
            success_tot = []
            for images, labels in tqdm(loaders['test'], total=len(loaders['test']), desc=attack_name):
                images, labels = images.to(device), labels.to(device)
                _, _, success = attack(fmodel, images, labels, epsilons=eps)
                success_tot.append(success)
            success_tot = torch.cat(success_tot, -1)

            # calculate and report the robust accuracy
            robust_accuracy = 1 - success_tot.float().mean(axis=-1)
            for epsilon, acc in zip(eps, robust_accuracy):
                res[curr_alpha][attack_name + '_' + str(epsilon)] = acc.item()

            pickle.dump(res, open(res_path, 'wb'))
        pickle.dump(res, open(res_path, 'wb'))
    pickle.dump(res, open(res_path, 'wb'))


def run_attacks_cleverhans(res_path):
    MORPH_MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results_morph_correct/CIFAR-10/densenet-82-8-8'
    MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results/CIFAR-10/densenet-82-8-8'
    # UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_2_1581641733617'
    # CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_2_1581641746832'
    # CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_2_1581641777871'
    UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_3_1585505819121'
    CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_3_1585505685528'
    CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_3_1585505042819'

    rel_dirs = [UNCORR_MODEL_DIR, CORR_MODEL_DIR, CORR_MODEL_DIR_2]
    alpha = ['0', '0.1', '0.2']

    res = dict.fromkeys(alpha)
    batch_size = 128  # 516
    n_workers = 20
    dataset = 'CIFAR-10'
    network = 'densenet-82-8-8'
    loaders, _ = get_dataloaders_(batch_size, 0, dataset, False, early_stop=False, n_workers=n_workers)
    n_models = 3

    params = {}
    params['densenet-82-8-8'] = {'num_modules': 2, 'bottleneck': True, 'reduction': 0.5, 'depth': 82, 'growth_rate': 8,
                                 'input_shape': (3, 32, 32), 'output_dim': 10}
    network = 'densenet-82-8-8'
    model = DenseNet(input_shape=params[network]['input_shape'],
                     output_dim=params[network]['output_dim'],
                     growth_rate=params[network]['growth_rate'],
                     depth=params[network]['depth'],
                     reduction=params[network]['reduction'],
                     bottleneck=params[network]['bottleneck'],
                     num_modules=n_models)

    device = torch.device("cuda")
    reports = dict.fromkeys(alpha)
    for model_path, curr_alpha in tqdm(zip(rel_dirs, alpha), total=len(alpha)):
        weight_path = path.join(MODEL_DIR, model_path, 'trial_0/0.0/weights/final_weights.pt')
        model.reset_parameters()
        model.load_state_dict(torch.load(weight_path))
        model.eval()  # model.train(mode=False)
        net = ModelMeanEP(model).to(device)

        report = dict()
        for x, y in tqdm(loaders['test'], total=len(loaders['test'])):
            x, y = x.to(device), y.to(device)
            report['nb_test'] = report.get('nb_test', 0) + y.size(0)

            _, y_pred = net(x).max(1)  # model prediction on clean examples
            report['acc'] = report.get('acc', 0) + y_pred.eq(y).sum().item()

            # model prediction on FGM adversarial examples
            x_adv = fast_gradient_method(net, x, 0.02, np.inf)
            _, y_pred = net(x_adv).max(1)  # model prediction on FGM adversarial examples
            report['FGM_0.02'] = report.get('FGM_0.02', 0) + y_pred.eq(y).sum().item()

            x_adv = fast_gradient_method(net, x, 0.04, np.inf)
            _, y_pred = net(x_adv).max(1)  # model prediction on FGM adversarial examples
            report['FGM_0.04'] = report.get('FGM_0.04', 0) + y_pred.eq(y).sum().item()

            # model prediction on BIM adversarial examples
            x_adv = projected_gradient_descent(net, x, eps=0.01, eps_iter=0.01 / 10, nb_iter=10, norm=np.inf, rand_init=0)
            _, y_pred = net(x_adv).max(1)
            report['BIM_0.01'] = report.get('BIM_0.01', 0) + y_pred.eq(y).sum().item()

            x_adv = projected_gradient_descent(net, x, eps=0.02, eps_iter=0.02 / 10, nb_iter=10, norm=np.inf, rand_init=0)
            _, y_pred = net(x_adv).max(1)
            report['BIM_0.02'] = report.get('BIM_0.02', 0) + y_pred.eq(y).sum().item()

            # model prediction on PGD adversarial examples
            x_adv = projected_gradient_descent(net, x, eps=0.01, eps_iter=0.01 / 10, nb_iter=10, norm=np.inf)
            _, y_pred = net(x_adv).max(1)
            report['PGD_0.01'] = report.get('PGD_0.01', 0) + y_pred.eq(y).sum().item()

            x_adv = projected_gradient_descent(net, x, eps=0.02, eps_iter=0.02 / 10, nb_iter=10, norm=np.inf)
            _, y_pred = net(x_adv).max(1)
            report['PGD_0.02'] = report.get('PGD_0.02', 0) + y_pred.eq(y).sum().item()

        for key in ['acc', 'FGM_0.02', 'FGM_0.04', 'BIM_0.01', 'BIM_0.02', 'PGD_0.01', 'PGD_0.02']:
            report[key] = (report[key] / report['nb_test']) * 100.

        reports[curr_alpha] = report
        pickle.dump(reports, open(res_path, 'wb'))
    pickle.dump(reports, open(res_path, 'wb'))


def get_TTR_FTR_curve(prob_prob, distractors_prob, prob_labels):
    open_set_1st_labels_0 = np.argmax(distractors_prob[0], 1)
    open_set_1st_labels_1 = np.argmax(distractors_prob[1], 1)
    open_set_1st_scores_0 = np.max(distractors_prob[0], 1)
    open_set_1st_scores_1 = np.max(distractors_prob[1], 1)

    mean_pred = np.argmax(prob_prob[-1], 1)
    mean_score = np.max(prob_prob[-1], 1)

    a = np.sum((open_set_1st_labels_0 == open_set_1st_labels_1))
    b = len(prob_labels)
    corr = (100.0*a / b)

    prev_FTR = -1
    prev_TTR = -1
    THs = []
    TTRs = []
    FTRs = []
    for i, TH in enumerate(np.arange(0, 1, 0.00001)):
        FTR = np.sum((open_set_1st_labels_0 == open_set_1st_labels_1) & (open_set_1st_scores_0 > TH) & (open_set_1st_scores_1 > TH)) / b
        TTR = np.sum((mean_score > TH) & (mean_pred == prob_labels)) / b

        if (prev_FTR != FTR and prev_TTR != TTR) or (i%100 == 0):
            prev_FTR = FTR
            prev_TTR = TTR
            THs.append(TH)
            TTRs.append(TTR)
            FTRs.append(FTR)
    return THs, TTRs, FTRs, corr


def ood_test(MODEL_DIR, res_path):
    rel_dirs = [x for x in os.listdir(MODEL_DIR) if '2020' in x]
    alpha = [re.findall('a=([0-9, \.]*)_', d)[0] for d in rel_dirs]
    learner = prep_learner()

    res_dir = dict.fromkeys(alpha)
    for model_path, curr_alpha in zip(rel_dirs, alpha):
        conf.save_path = pathlib.Path(path.join(MODEL_DIR, model_path))
        fix_str = [x for x in os.listdir(path.join(MODEL_DIR, model_path)) if '2020' in x][0][8:]
        learner.load_state(conf, fix_str, model_only=True, from_save_folder=True)

        # distractors
        set_distractors(learner)
        distractors_prob, distractors_predictions, distractors_labels = get_evaluation(learner)

        # probs
        set_probes(learner)
        prob_prob, prob_predictions, prob_labels = get_evaluation(learner)

        THs, TTRs, FTRs, corr = get_TTR_FTR_curve(prob_prob, distractors_prob, prob_labels)
        print(curr_alpha, corr)
        res_dir[curr_alpha] = [THs, TTRs, FTRs, corr]
    pickle.dump(res_dir, open(res_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for CBIS-DDSM')
    parser.add_argument("-ood", "--ood_test", help="to ood test instead?", default=0, type=int)

    args = parser.parse_args()
    conf = get_config()

    if args.ood_test:
        res_path = str('cifar_ood_res.pkl')
        ood_test(res_path)
    else:
        res_path = str('cifar_attack_res.pkl')
        run_attacks_cleverhans(res_path)
        #run_attacks(res_path)
