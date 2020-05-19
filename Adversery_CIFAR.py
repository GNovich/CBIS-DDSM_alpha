from config import get_config
import argparse
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
from foolbox.attacks import FGSM, L2BasicIterativeAttack as BIM, PGD, \
                            L2CarliniWagnerAttack as CaW, EADAttack as EAD, \
                            L2BrendelBethgeAttack as MIM_maybe
sys.path.append('/mnt/md0/orville/Miriam/modular-loss-experiments-morph/')
from src.models import DenseModel, ConvModel, DenseNet
from src.argument_parser import parse_args
from src.dataloader import get_dataloaders
from src.distributions import distributions
class_dir = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
             5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# Momentum Iterative Method (MIM)
# Jacobian-based Saliency Map Attack (JSMA)

attack_list = [
    (FGSM, [0.02, 0.04]),
    (partial(BIM, rel_stepsize=0.1, steps=10), [0.01, 0.02]),
    (partial(PGD, rel_stepsize=0.1, steps=10), [0.01, 0.02]),
    (partial(CaW, steps=1000, stepsize=0.01, confidence=0.001), [0.001]),
    (partial(CaW, steps=1000, stepsize=0.01, confidence=0.01), [0.01]),
    (partial(CaW, steps=1000, stepsize=0.01, confidence=0.1), [0.1]),
    (partial(EAD, steps=1000, initial_stepsize=0.01, confidence=1), [1]),
    (partial(EAD, steps=1000, initial_stepsize=0.01, confidence=5), [5])
    #(partial(MIM_maybe, lr=0.01), [0.01, 0.02]),
]

attack_list_names = [
    'FGSM',
    'BIM',
    'PGD',
    'CaW',
    'CaW',
    'CaW',
    'EAD',
    'EAD'
    #'(MIM, 0.01)',
    #'(MIM, 0.02)',
]
"""
# debug
attack_list = [
    (FGSM, [0, 0.02, 0.04, 0.2, 0.4]),
]

attack_list_names = [
    'FGSM',
]
"""
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
        return torch.cat([x.mean(0) for x in torch.chunk(res, len(self.model.device_ids))])
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


def run_OnePixleAttack(res_path, model_num=-1, n_pixel=1, ncl=False):
    if ncl:
        MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results_ncl/CIFAR-10/densenet-82-8-8'
        rel_dirs = ['alpha_0.0_gamma_0.02_n_models_2_1583114412120',
                    'alpha_0.0_gamma_0.05_n_models_2_1583114439810']
        alpha = ['0.02', '0.05']
        res_path = res_path + '_ncl'
    else:
        MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results/CIFAR-10/densenet-82-8-8'
        # UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_2_1581641733617'
        # CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_2_1581641746832'
        # CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_2_1581641777871'
        # UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_3_1585505819121'
        # CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_3_1585505685528'
        # CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_3_1585505042819'
        # rel_dirs = [UNCORR_MODEL_DIR, CORR_MODEL_DIR, CORR_MODEL_DIR_2]
        rel_dirs = ['alpha_0.0_gamma_0.0_n_models_3_1585505819121',
                  'alpha_0.1_gamma_0.0_n_models_3_1589795142450',
                  'alpha_0.2_gamma_0.0_n_models_3_1589794987034',
                  'alpha_0.3_gamma_0.0_n_models_3_1589795486214',
                  'alpha_0.4_gamma_0.0_n_models_3_1589796192038',
                  'alpha_0.5_gamma_0.0_n_models_3_1589796200262',
                  'alpha_0.6_gamma_0.0_n_models_3_1589796218204',
                  'alpha_0.7_gamma_0.0_n_models_3_1589796234665']
        alpha = list(map(lambda x: format(x, '2.1f'), np.arange(0.0, 0.8, 0.1)))

    if model_num!=-1:
        rel_dirs = [rel_dirs[model_num]]
        alpha = [alpha[model_num]]
        res_path = res_path+"_"+str(model_num)

    batch_size = 1  # 516
    n_workers = 20
    dataset = 'CIFAR-10'
    network = 'densenet-82-8-8'
    loaders, _ = get_dataloaders_(batch_size, 0, dataset, False, early_stop=False, n_workers=n_workers)
    n_models = 3 if not ncl else 2

    params = {}
    params['densenet-82-8-8'] = {'num_modules': n_models, 'bottleneck': True, 'reduction': 0.5, 'depth': 82, 'growth_rate': 8,
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
        if ncl:
            weight_path = path.join(MODEL_DIR, model_path, 'trial_0/' + curr_alpha + '/weights/final_weights.pt')
        else:
            weight_path = path.join(MODEL_DIR, model_path, 'trial_0/0.0/weights/final_weights.pt')

        model.reset_parameters()
        model.load_state_dict(torch.load(weight_path))
        model.eval()  # model.train(mode=False)
        net = ModelMeanEP(model).to(device)
        results = OnePixleAttack(net, loaders['test'], pixels=n_pixel)
        # (pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False)

        reports[curr_alpha] = results
        pickle.dump(reports, open(res_path, 'wb'))
    pickle.dump(reports, open(res_path, 'wb'))


from absl import app, flags
from easydict import EasyDict
from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent
FLAGS = flags.FLAGS
def run_attacks_cleverhans(res_path, ncl=False):
    if ncl:
        MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results_ncl/CIFAR-10/densenet-82-8-8'
        rel_dirs = ['alpha_0.0_gamma_0.02_n_models_2_1583114412120',
                    'alpha_0.0_gamma_0.05_n_models_2_1583114439810']
        alpha = ['0.02', '0.05']
        res_path = res_path + '_ncl'
    else:
        MODEL_DIR = '/mnt/md0/orville/Miriam/modular-loss-experiments-morph/results/CIFAR-10/densenet-82-8-8'
        # UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_2_1581641733617'
        # CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_2_1581641746832'
        # CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_2_1581641777871'
        # UNCORR_MODEL_DIR = 'alpha_0.0_gamma_0.0_n_models_3_1585505819121'
        # CORR_MODEL_DIR = 'alpha_0.1_gamma_0.0_n_models_3_1585505685528'
        # CORR_MODEL_DIR_2 = 'alpha_0.2_gamma_0.0_n_models_3_1585505042819'
        # rel_dirs = [UNCORR_MODEL_DIR, CORR_MODEL_DIR, CORR_MODEL_DIR_2]
        rel_dirs = ['alpha_0.0_gamma_0.0_n_models_3_1585505819121',
                  'alpha_0.1_gamma_0.0_n_models_3_1589795142450',
                  'alpha_0.2_gamma_0.0_n_models_3_1589794987034',
                  'alpha_0.3_gamma_0.0_n_models_3_1589795486214',
                  'alpha_0.4_gamma_0.0_n_models_3_1589796192038',
                  'alpha_0.5_gamma_0.0_n_models_3_1589796200262',
                  'alpha_0.6_gamma_0.0_n_models_3_1589796218204',
                  'alpha_0.7_gamma_0.0_n_models_3_1589796234665']
        alpha = list(map(lambda x: format(x, '2.1f'), np.arange(0.0, 0.8, 0.1)))

    batch_size = 256  # 128  # 516
    n_workers = 20
    dataset = 'CIFAR-10'
    network = 'densenet-82-8-8'
    loaders, _ = get_dataloaders_(batch_size, 0, dataset, False, early_stop=False, n_workers=n_workers)
    n_models = 2 if ncl else 3

    params = {}
    params['densenet-82-8-8'] = {'num_modules': n_models, 'bottleneck': True, 'reduction': 0.5, 'depth': 82, 'growth_rate': 8,
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
        if ncl:
            weight_path = path.join(MODEL_DIR, model_path, 'trial_0/' + curr_alpha + '/weights/final_weights.pt')
        else:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for CBIS-DDSM')
    parser.add_argument("-ood", "--ood_test", help="do ood test instead?", default=0, type=int)
    parser.add_argument("-opa", "--one_pixel", help="do one pixel attack instead?", default=0, type=int)
    parser.add_argument("-model_num", "--model_num", help="run a spesific model?", default=-1, type=int)
    parser.add_argument("-n_pixel", "--n_pixel", help="n_pixel?", default=1, type=int)
    parser.add_argument("-ncl", "--ncl", help="ncl", default=0, type=int)

    args = parser.parse_args()
    conf = get_config()

    if args.one_pixel:
        res_path = str('cifar_one_pixle_attack_res.pkl')
        run_OnePixleAttack(res_path, args.model_num, args.n_pixel, args.ncl)
    else:
        res_path = str('cifar_attack_res.pkl')
        run_attacks_cleverhans(res_path, ncl=args.ncl)
        #run_attacks(res_path)
