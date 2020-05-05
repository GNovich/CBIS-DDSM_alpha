from config import get_config
from Learner import Learner
from PatchLearner import PatchLearner, PatchLearnerMult
import argparse
import torch
from functools import partial
from torch.nn import MSELoss
from Pearson import pearson_corr_loss, ncl_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for CBIS-DDSM')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="number of workers", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="databases: [crop_data, whole_data]", default='crop_data', type=str)
    parser.add_argument("-s", "--epoch_per_save", help="save_every s epochs", default=1, type=int)
    parser.add_argument("-net", "--net_mode", help="choose net", default='resnet50', type=str)

    parser.add_argument("-n", "--n_models", help="how many duplicate nets to use. 1 leads to basic training, "
                                                 "making -a and -p flags redundant", default=1, type=int)
    parser.add_argument("-patch", "--n_patch", help="how many patches per image", default=2, type=int)
    parser.add_argument("-bkg", "--bkg_prob", help="how many patches per image", default=.5, type=float)
    parser.add_argument("-mul", "--mul", help="use mult mode?", default=0, type=int)
    parser.add_argument("-rank", "--local_rank", help="rank for mul", default=0, type=int)
    parser.add_argument("-pre", "--pre_train", help="use a pretrain net?", default=0, type=int)
    parser.add_argument("-pre_layers", "--pre_layers", help="layer steps to use?", default=[], type=int, nargs='*')
    parser.add_argument("-pre_step", "--pre_steps", help="what steps to use?", default=[3, 10, 37], type=int, nargs='*')
    parser.add_argument("-ngpu", "--ngpu", help="how many gpu's to use?", default=1, type=int)
    parser.add_argument("-half", "--half", help="use half precisions?", default=0, type=int)
    parser.add_argument("-no_bkg", "--no_bkg", help="4 class mode", default=0, type=int)
    parser.add_argument("-cancer", "--cancer_only", help="limit label to benign/malignant", default=0, type=int)
    parser.add_argument("-type", "--type_only", help="limit label to mass/calcification", default=0, type=int)

    # TODO maybe add option to specify a network mix instead of duplicates
    parser.add_argument("-m", "--milestones", help="fractions of where lr will be tuned", default=[], type=int, nargs='*')
    parser.add_argument("-a", "--alpha", help="balancing parameter", default=0, type=float)
    parser.add_argument("-t", "--sig_thresh", help="thresholding of the most correct class", default=0.9, type=float)
    parser.add_argument("-p", "--pearson", help="using pearson loss", default=False, type=bool)
    parser.add_argument("-ncl", "--ncl", help="using Negative Correlation Loss", default=False, type=bool)
    parser.add_argument("-mean", "--joint_mean", help="using mean loss", default=False, type=bool)
    parser.add_argument("-morph_dir", "--morph_dir", help="use a morph directory", default='', type=str)
    parser.add_argument("-morph_a", "--morph_alpha", help="balance parameter", default=10., type=float)

    parser.add_argument("-c", "--cpu_mode", help="force cpu mode", default=0, type=int)

    args = parser.parse_args()
    conf = get_config()

    # training param
    assert not (args.cancer_only and args.type_only)  # choose at most one
    conf.cancer_only = args.cancer_only
    conf.type_only = args.type_only
    conf.no_bkg = args.no_bkg
    conf.half = args.half
    conf.ngpu = args.ngpu
    conf.pre_layers = args.pre_layers
    conf.pre_steps = args.pre_steps
    conf.pre_train = args.pre_train
    conf.local_rank = args.local_rank
    conf.n_patch = args.n_patch
    conf.bkg_prob = args.bkg_prob if not args.no_bkg else 0
    conf.net_mode = args.net_mode
    conf.evaluate_every = 3  # TODO see if relevant
    conf.epoch_per_save = args.epoch_per_save
    conf.data_mode = args.data_mode
    conf.cpu_mode = args.cpu_mode
    conf.device = torch.device("cuda" if (torch.cuda.is_available() and not conf.cpu_mode) else "cpu")
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.epochs = args.epochs
    conf.milestones = args.milestones

    # pearson param
    conf.alpha = args.alpha
    conf.sig_thresh = args.sig_thresh
    conf.n_models = args.n_models
    conf.pearson = args.pearson
    conf.joint_mean = args.joint_mean
    conf.ncl = args.ncl

    # morph param
    conf.morph_alpha = args.morph_alpha
    conf.morph_dir = args.morph_dir

    # loss funcs
    conf.pearson_loss = partial(pearson_corr_loss, threshold=conf.sig_thresh)
    conf.ncl_loss = partial(ncl_loss)
    conf.morph_loss = MSELoss()

    # create learner and go
    param_desc = '_'.join([
        str(conf.net_mode), 'lr='+str(conf.lr), 'm='+'_'.join([str(m) for m in conf.milestones]),
        ('a='+str(conf.alpha) if conf.n_models>1 else ''),
        str(conf.batch_size), str(conf.n_patch), 'pre' if conf.pre_train else ''])
    conf.log_path = str(conf.log_path) + '_' + param_desc
    conf.save_path = str(conf.save_path) + '_' + param_desc

    learner = PatchLearnerMult(conf) if args.mul else PatchLearner(conf)
    # face_learner(conf) if conf.n_models == 1 else face_learner_corr(conf)

    if conf.pre_train:
        learner.pretrain(conf)
    else:
        learner.train(conf, conf.epochs)
