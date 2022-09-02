#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os
import time, datetime

def str2bool(v):
    #https://eehoeskrap.tistory.com/521
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getTimestamp():
    utc_timestamp = int(time.time())
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime("%Y_%m_%d_%H_%M_%S")
    return date

def args_parser():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model',            type=str,       default='resnet18',  help='resnet18|resnet50')
    parser.add_argument('--pretrained',       type=str2bool,  default=False,       help='pretrained backbone')
    parser.add_argument('--num_classes',      type=int,       default=10,          help="number of classes")
    parser.add_argument('--group_norm',       type=str2bool,  default=False,       help="group normalization")
    
    # Experimental setup
    parser.add_argument("--exp",              type=str,       default="FLSL",      choices=["simclr", "simsiam", "centralized", "FLSL"])
    parser.add_argument("--wandb_tag",        type=str,       default="",        help="optional tag for wandb logging")
    parser.add_argument("--alpha",            type=float,     default=0.5,       help="dirichlet param 0<alpha<infty controls iidness alpha=0:non iid")
    
    # SimCLR
    parser.add_argument("--temperature",      type=float,     default=0.1,       help="softmax temperature")
    parser.add_argument('--n_views',          type=int,       default=2,         help="default simclr n_views=2")

    # SimSiam
    parser.add_argument("--pred_dim",         type=int,       default=256,       help="pred dim for simsiam")

    # SimCLR & SimSiam 
    parser.add_argument('--strength',         type=float,     default=0.5,       help="augmentation strength 0<s<1")
    parser.add_argument('--target_size',      type=int,       default=32,        help="augmentation target width (=height)")
    parser.add_argument('--freeze',           type=str2bool,  default=False,      help='freeze feature extractor during linear eval')
    parser.add_argument('--out_dim',          type=int,       default=512,       help="output dimension of the feature for simclr and simsiam")
    
    # FL
    parser.add_argument("--num_users",      type=int,      default=10,        help="num users")
    parser.add_argument("--num_items",      type=int,      default=32,        help="num data each client holds")
    parser.add_argument("--iid",            type=str2bool, default=True,      help="iid on clients")
    parser.add_argument('--epochs',         type=int,      default=200,        help="number of rounds of training") # FedMatch
    parser.add_argument('--frac',           type=float,    default=1,       help='the fraction of clients: C')
    parser.add_argument('--local_ep',       type=int,      default=10,         help="the number of local epochs: E")
    parser.add_argument('--local_bs',       type=int,      default=32,         help="local batch size")
    parser.add_argument('--lr',             type=float,    default=0.001,      help='learning rate')
    parser.add_argument('--momentum',       type=float,    default=0.9,        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay',   type=float,    default=1e-4,       help='weight decay (default: 1e-4)')
    parser.add_argument('--dataset',        type=str,       default='cifar',  choices=["mnist", "cifar"],              help="mnist|cifar")
    parser.add_argument('--optimizer',      type=str,       default='adam',               help="type of optimizer")
    
    # FedProx
    parser.add_argument('--mu',             type=float,     default=0.01)

    # Train setting
    parser.add_argument("--parallel",       type=str2bool,  default=True,                help="parallel training with threads")
    parser.add_argument("--num_workers",    type=int,       default=8,                    help="num workers for dataloader")
    parser.add_argument('--log_path',       type=str,       default='./logs',             help="tensorboard log dir")
    parser.add_argument('--seed',           type=int,       default=2022,                 help='random seed')
    parser.add_argument('--ckpt_path',      type=str,       default="./checkpoints/checkpoint.pth.tar", help="model ckpt save path")
    parser.add_argument('--data_path',      type=str,       default="./data",             help="path to dataset")
    parser.add_argument('--train_device',   type=str,       default="0",                  help="gpu device number for train")
    parser.add_argument('--finetune_epoch', type=int,       default=1,                   help='finetune epochs at server')
    parser.add_argument('--finetune',       type=str2bool,  default=True,                 help="finetune at the server")
    parser.add_argument("--ckpt_criterion",  type=str,     default="loss", help="ckpt criterion loss|top1")
    parser.add_argument("--agg",             type=str,     default="fedavg", choices=["fedavg", "fedprox", "fedmatch"])

    args = parser.parse_args() 
    
    # if no experiment is specified on path 
    if not args.log_path.split("logs")[-1]: 
        args.log_path = os.path.join(args.log_path, getTimestamp())
    if not args.finetune:
        args.finetune_epoch = 1


    if args.exp == "FLSL":
        get_FLSL_opts(args, args.iid)

    elif args.exp == "centralized":
        get_centralized_opts(args, args.iid)
    
    elif args.exp == "simclr":
        get_simclr_opts(args, args.iid)

    elif args.exp == "simsiam":
        get_simsiam_opts(args, args.iid)

    #############TODO My method#################

    return args

def get_centralized_opts(args, iid = False):
    if iid == True:
        args.alpha = 100000 # arbitrary large number for iid

    else:
        args.alpha = 0.5    # Non-i.i.d. 
    
    args.num_users = 100
    args.num_items = 300
    args.epochs  = 100
    args.frac = 0.1
    args.local_ep = 10
    args.local_bs = 16
    args.finetune = False
    args.freeze = False

def get_FLSL_opts(args, iid = False):
    if iid == True:
        args.alpha = 100000

    else:
        args.alpha = 0.1

    args.num_users = 100
    args.num_items = 300
    args.epochs = 100
    args.frac = 0.1
    args.local_ep = 10
    args.local_bs = 16
    # aggregate only 
    args.finetune = False
    args.freeze = False

def get_simclr_opts(args, iid = False):
    if iid == True:
        args.alpha = 100000
    else:
        args.alpha = 0.5

    args.num_users = 100
    args.num_items = 300
    args.epochs = 100
    args.frac = 0.1
    args.local_ep = 10
    args.local_bs = 16
    # aggregate and train linear at server (finetune) 
    args.finetune = True
    
def get_simsiam_opts(args, iid = False):
    if iid == True:
        args.alpha = 100000
    else:
        args.alpha = 0.5

    args.num_users = 100
    args.num_items = 300
    args.epochs = 100
    args.frac = 0.1
    args.local_ep = 10
    args.local_bs = 16
    args.finetune = True

