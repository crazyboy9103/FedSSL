
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate

import numpy as np

class AverageMeter():
    def __init__(self, name):
        self.name = name
        self.values = []
    
    def update(self, value):
        self.values.append(value)
        
    def get_result(self):
        return sum(self.values)/len(self.values)
    
    def reset(self):
        self.values = []
        
def get_train_idxs(dataset, num_users, num_items, alpha):
    labels = dataset.targets
    
    # Collect idxs for each label
    idxs_labels = {i: set() for i in range(10)}
    for idx, label in enumerate(labels):
        idxs_labels[label].add(idx)
    

    # 10 labels
    class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    class_dist = (class_dist * num_items).astype(int)
    
    if num_users == 1:
        class_dist = class_dist
        
        for _class, class_num in enumerate(class_dist[0]):
            if class_num > len(idxs_labels[_class]):
                class_dist[0][_class] = len(idxs_labels[_class])
            
    else:
            
        for _class, class_num in enumerate(class_dist.T.sum(axis=1)):
            assert class_num < len(idxs_labels[_class]), "num_items must be smaller"
    
    
    dict_users = {i: set() for i in range(num_users)}
    dists = {i: [0 for j in range(10)] for i in range(num_users)}
    
    for client_id, client_dist in enumerate(class_dist):
        for _class, num in enumerate(client_dist):
            sample_idxs = idxs_labels[_class]
            dists[client_id][_class] += num
            
            sampled_idxs = set(np.random.choice(list(sample_idxs), size=num, replace=False)) 
            # accumulate
            dict_users[client_id].update(sampled_idxs)
            
            # exclude assigned idxs
            idxs_labels[_class] = sample_idxs - sampled_idxs
            
    for i, data_idxs in dict_users.items():
        dict_users[i] = list(data_idxs)
    

    
    # for client_id, dist in dists.items():
    #     plt.figure(client_id)
    #     plt.title(f"client {client_id} class distribution")
    #     plt.xlabel("class")
    #     plt.ylabel("num items")
    #     plt.bar(range(10), dist, label=client_id)
    #     plt.savefig(f"./alpha/client_{client_id}_{sum(dist)}_{alpha}_{num_users}.png")
    #     plt.clf()
    
    
    return dict_users


class SimCLRTransformWrapper(object):
    def __init__(self, base_transform, args):
        self.base_transform = base_transform
        self.n_views = args.n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)] # two views by default

    
def get_dataset(args):
    cifar_data_path = os.path.join(args.data_path, "cifar")
    mnist_data_path = os.path.join(args.data_path, "mnist")
    # transforms set according to https://github.com/guobbin/PFL-MoE/blob/master/main_fed.py
    if args.exp == "simclr" or args.exp == "simsiam":
        s = args.strength
        target_size = args.target_size
        
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        train_transforms = transforms.Compose([
            #transforms.RandomResizedCrop(size=target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * target_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transforms = transforms.Compose([
            #transforms.Resize(size=target_size), 
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    # Normal FL
    elif args.exp == "FL":
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    
    if args.dataset == "cifar":
        data = datasets.CIFAR10
        data_path = cifar_data_path

    elif args.dataset == "mnist":
        data = datasets.MNIST
        data_path = mnist_data_path
        
    train_dataset = data(
        data_path, 
        train=True, 
        transform=SimCLRTransformWrapper(train_transforms, args) if args.exp != "FL" else train_transforms, 
        download=True
    )

    test_dataset = data(
        data_path, 
        train=False, 
        transform=test_transforms, 
        download=True
    )  
        
    user_train_idxs = get_train_idxs(
        train_dataset, 
        args.num_users, 
        args.num_items,
        args.alpha
    )
            
    return train_dataset, test_dataset, user_train_idxs


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0]) # this would be on cuda:0 
    for key in w_avg.keys():
        for i in range(1, len(w)):
            if w_avg[key].get_device() != w[i][key].get_device():
                w[i][key] = w[i][key].to(w_avg[key].get_device())
            w_avg[key] += w[i][key]
            
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args, writer):
    print('Experimental details:')
    print(f'    Seed            : {args.seed}')
    print(f'    Dataset         : {args.dataset}')
    print(f'    Model           : {args.model}')
    print(f'    Pretrained      : {args.pretrained}')
    print(f'    Optimizer       : {args.optimizer}')
    print(f'    Learning rate   : {args.lr}')
    print(f'    Total Rounds    : {args.epochs}')
    print(f'    Alpha           : {args.alpha}')
    print(f'    Momentum        : {args.momentum}')
    print(f'    Weight decay    : {args.weight_decay}')
    print(f'    Sup Warmup      : {args.sup_warmup}')
    print(f'    Server data frac: {args.server_data_frac}')
    
    if args.exp == "simclr":
        print("SimCLR")
        print(f'    Warmup          : {args.warmup}')
        print(f'    Freeze          : {args.freeze}')
        print(f'    Adapt Epochs    : {args.adapt_epoch}')
        print(f'    Warmup Epochs   : {args.warmup_epochs}')
        print(f'    Warmup Batchsize: {args.warmup_bs}')
        print(f'    Temperature     : {args.temperature}')
        print(f'    Output dim      : {args.out_dim}')
        print(f'    N views         : {args.n_views}')


    elif args.exp == "simsiam":
        print("SimSiam")
        print(f'    Warmup          : {args.warmup}')
        print(f'    Freeze          : {args.freeze}')
        print(f'    Adapt Epochs    : {args.adapt_epoch}')
        print(f'    Warmup Epochs   : {args.warmup_epochs}')
        print(f'    Warmup Batchsize: {args.warmup_bs}')
        print(f'    Output dim      : {args.out_dim}')
        print(f'    Pred   dim      : {args.pred_dim}')
        

    
    elif args.exp == "FL":
        print("FL")
        print(f'    Warmup          : {args.warmup}')
        print(f'    Freeze          : {args.freeze}')
        print(f'    Adapt Epochs    : {args.adapt_epoch}')
        print(f'    Warmup Epochs   : {args.warmup_epochs}')
        print(f'    Warmup Batchsize: {args.warmup_bs}')
        
        
        
    print('Federated parameters:')
    print(f'    Number of users                : {args.num_users}')
    print(f'    Fraction of users              : {args.frac}')
    print(f'    Number of train items per user : {args.num_items}')
    print(f'    Local Batch size               : {args.local_bs}')
    print(f'    Local Epochs                   : {args.local_ep}')
    print(f'    Checkpoint path                : {args.ckpt_path}')
    print(f'    Tensorboard log path           : {args.log_path}')

    
    
    