from config import get_config
from Learner import Learner
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
    parser.add_argument("-d", "--data_mode", help="databases: [crop, whole]", default='crop', type=str)
    parser.add_argument("-s", "--save_per_epoch", help="num of times to save per epoch", default=1, type=int)

    # ganovich - added some parameters
    parser.add_argument("-n", "--n_models", help="how many duplicate nets to use. 1 leads to basic training, "
                                                 "making -a and -p flags redundant", default=1, type=int)

    # TODO maybe add option to specify a network mix instead of duplicates
    parser.add_argument("-m", "--milestones", help="epoch list where lr will be tuned", default=[12, 15, 18], type=int, nargs='*')
    parser.add_argument("-a", "--alpha", help="balancing parameter", default=0, type=float)
    parser.add_argument("-t", "--sig_thresh", help="thresholding of the most correct class", default=0.9, type=float)
    parser.add_argument("-p", "--pearson", help="using pearson loss", default=False, type=bool)
    parser.add_argument("-ncl", "--ncl", help="using Negative Correlation Loss", default=False, type=bool)
    parser.add_argument("-mean", "--joint_mean", help="using mean loss", default=False, type=bool)
    parser.add_argument("-morph_dir", "--morph_dir", help="use a morph directory", default='', type=str)
    parser.add_argument("-morph_a", "--morph_alpha", help="balance parameter", default=10., type=float)

    parser.add_argument("-c", "--cpu_mode", help="force cpu mode", default=False, type=bool)

    args = parser.parse_args()
    conf = get_config()

    # training param

    conf.save_per_epoch = args.save_per_epoch
    conf.data_mode = args.data_mode
    conf.cpu_mode = args.cpu_mode
    conf.device = torch.device("cuda:0" if (torch.cuda.is_available() and not conf.cpu_mode) else "cpu")
    conf.lr = args.lr
    conf.milestones = args.milestones
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.save_per_epoch = 3

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
    learner = Learner(conf)
    # face_learner(conf) if conf.n_models == 1 else face_learner_corr(conf)
    learner.train(conf, args.epochs)
