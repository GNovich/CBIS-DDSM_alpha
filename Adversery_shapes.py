from config import get_config
import argparse
from ShapeLearner import ShapeLearner
from ShapeLoader import ShapeDataSet
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import pickle
import torch
from os import path
from pathlib import Path
import os
from itertools import product
import re
from tqdm import tqdm
from functools import partial
import eagerpy as ep
from foolbox import PyTorchModel
from foolbox.attacks import FGSM, L2BasicIterativeAttack as BIM, PGD, \
                            L2CarliniWagnerAttack as CaW, EADAttack as EAD, \
                            L2BrendelBethgeAttack as MIM_maybe

# Momentum Iterative Method (MIM)
# Jacobian-based Saliency Map Attack (JSMA)

attack_list = [
    (FGSM, [0.002, 0.004]),
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


class JointModelEP(torch.nn.Module):
    def __init__(self, models, device=None):
        super(JointModelEP, self).__init__()
        self.models = models
        self.bounds = models[0].bounds

    def forward(self, x):
        res = []
        for model in self.models:
            res.append(model(x))
        return ep.stack(res, 1).mean(1)

def run_attacks(MODEL_DIR, res_path):
    rel_dirs = os.listdir(MODEL_DIR)
    alpha = [re.findall('a=([0-9, \.]*)_', d)[0] for d in rel_dirs if '2020' in d]
    res = dict.fromkeys(alpha)
    learner = prep_learner()

    for model_path, curr_alpha in tqdm(zip(rel_dirs, alpha), total=len(alpha)):
        conf.save_path = Path(path.join(MODEL_DIR, model_path))
        fix_str = [x for x in os.listdir(path.join(MODEL_DIR, model_path)) if 'model' in x][0][8:]
        learner.load_state(conf, fix_str, model_only=True, from_save_folder=True)

        # probs
        set_probes(learner)

        for model in learner.models:
            model = torch.nn.DataParallel(model.cuda(), device_ids=list(range(4)))
            model.eval()

        res[curr_alpha] = dict()
        for (attack, eps), attack_name in tqdm(zip(attack_list, attack_list_names),
                                                        desc='attaking ' + str(curr_alpha), total=len(attack_list)):
            fmodel = JointModelEP([PyTorchModel(m, bounds=(0, 1)) for m in learner.models], 'cuda')
            attack = attack()
            success_tot = []
            for images, labels in tqdm(learner.eval_loader, total=len(learner.eval_loader), desc=attack_name):
                images, labels = ep.astensors(images.to('cuda'), labels.to('cuda'))
                _, _, success = attack(fmodel, images, labels, epsilons=eps)
                success_tot.append(success)
            success_tot = ep.concatenate(success_tot, -1)

            # calculate and report the robust accuracy
            robust_accuracy = 1 - success.float32().mean(axis=-1)
            for epsilon, acc in zip(eps, robust_accuracy):
                res[curr_alpha][attack_name + '_' + str(epsilon)] = acc.item()

            pickle.dump(res, open(res_path, 'wb'))
        pickle.dump(res, open(res_path, 'wb'))


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
    rel_dirs = os.listdir(MODEL_DIR)
    alpha = [re.findall('a=([0-9, \.]*)_', d)[0] for d in rel_dirs if 'model' in d]
    learner = prep_learner()

    res_dir = dict.fromkeys(alpha)
    for model_path, curr_alpha in zip(rel_dirs, alpha):
        conf.save_path = Path(path.join(MODEL_DIR, model_path))
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
    parser.add_argument("-MODEL_DIR", "--MODEL_DIR", help="modeling_dir", default='', type=str)
    parser.add_argument("-ood", "--ood_test", help="to ood test instead?", default=0, type=int)

    args = parser.parse_args()
    conf = get_config()

    if args.ood_test:
        res_path = str(Path(args.MODEL_DIR) / 'ood_res.pkl')
        ood_test(args.MODEL_DIR, res_path)
    else:
        res_path = str(Path(args.MODEL_DIR) / 'attack_res.pkl')
        run_attacks(args.MODEL_DIR, res_path)