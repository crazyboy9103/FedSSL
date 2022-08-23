
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
import math

class CudaCKA:
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

# def evaluate_CKA_by_layer(local_weights_copy, global_model_copy):
#     global_model_copy.set_mode("linear")

#     noise = torch.rand(1, 3, 32, 32, device="cuda:0", requires_grad=False)
#     def flatten(tensor):
#         _, c, h, w = tensor.shape
#         return tensor.squeeze(0).view(c, h * w)


#     output_vectors = {}
#     for client_id, weight in local_weights_copy.items():
#         global_model_copy.load_state_dict(weight)
#         outputs = []
#         with torch.no_grad():
#             out = global_model_copy.conv1(noise)
#             out = global_model_copy.bn1(out)
#             out = global_model_copy.relu(out)
#             # head layer (conv1 + bn1 + relu + maxpool) output
#             output_head = global_model_copy.maxpool(out)
#             outputs.append(output_head.detach())

#             layers = [global_model_copy.layer1, global_model_copy.layer2, global_model_copy.layer3, global_model_copy.layer4]
#             out = output_head
#             for layer in layers:
#                 out = layer[0](out)
#                 outputs.append(out.detach())
#                 out = layer[1](out)
#                 outputs.append(out.detach())

#         outputs = list(map(lambda tensor: flatten(tensor),  outputs))
#         output_vectors[client_id] = outputs
    
#     for i in range(len(output_vectors)):
#         for j in range(i+1, len(output_vectors)):
#             vec

def get_length_gradients(local_weights_copy, global_model_copy):
    global_state_dict = copy.deepcopy(global_model_copy.state_dict())

    grads = []
    for client_id, state_dict in local_weights_copy.items():
        client_grad = []
        for key, weight in global_state_dict.items():
            if "weight" in key:
                l2_norm = torch.linalg.norm(state_dict[key].cpu() - weight.cpu()).detach().item()
                client_grad.append(l2_norm)
                print("l2norm", l2_norm)
        grad_sum = sum(client_grad)
        print(grad_sum)
        client_grad = [grad / grad_sum for grad in client_grad] # normalize
        grads.append(client_grad)
    print(np.array(grads).shape)
    means = np.array(grads).T.mean(1).tolist()
    print("means", means)
    return means


def feed_noise_to_models(local_weights_copy, global_model_copy, batch_size):
    # global_model_copy.set_mode("linear")

    noise = torch.rand(batch_size, 3, 32, 32, device="cuda:0", requires_grad=False)

    out_vectors = None
    for client_id, weight in local_weights_copy.items():
        global_model_copy.load_state_dict(weight)
        with torch.no_grad():
            out = global_model_copy.backbone(noise)
            if hasattr(global_model_copy, "projector"):
                out = global_model_copy.projector(out)

        if out_vectors == None:
            out_vectors = out[0].unsqueeze(0)
        
        else:
            out_vectors = torch.cat((out_vectors, out[0].unsqueeze(0)), dim=0)
    # print(out_vectors.shape)
    return out_vectors





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
    if args.exp in ["simclr", "simsiam"]:
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
        
    # Normal FL or centralized
    elif args.exp in ["FLSL", "centralized"]:
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
        transform=SimCLRTransformWrapper(train_transforms, args) if args.exp in ["simclr", "simsiam"] else train_transforms, 
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
    w_avg = copy.deepcopy(w[0]) 
    for key in w_avg.keys():
        for i in range(1, len(w)):
            if w_avg[key].get_device() != w[i][key].get_device():
                w[i][key] = w[i][key].to(w_avg[key].get_device())
            w_avg[key] += w[i][key]
            
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class CheckpointManager():
    def __init__(self, type):
        self.type = type
        if type == "loss":
            self.best_loss = 1E27 
        
        elif type == "top1":
            self.best_top1 = -1E27 

    def _check_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        
        return False
    
    def _check_top1(self, top1):
        if top1 > self.best_top1:
            self.best_top1 = top1
            return True
        
        return False


    def save(self, loss, top1, model_state_dict, checkpoint_path):
        save_dict = {
            "model_state_dict": model_state_dict, 
            # "optim_state_dict": optim_state_dict, 
            "loss": loss, 
            "top1": top1
        }
        if self.type == "loss" and self._check_loss(loss):
            torch.save(save_dict, checkpoint_path)

        elif self.type == "top1" and self._check_top1(top1):
            torch.save(save_dict, checkpoint_path)

        print(f"model saved at {checkpoint_path}")

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

    
    
    
