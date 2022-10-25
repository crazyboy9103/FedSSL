#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
from torchvision import models
import torch
import torch.nn.functional as F

class ResNet_model(nn.Module):
    def __init__(self, args):
        super(ResNet_model, self).__init__()
        self.args = args
        
        out_dim = args.out_dim 
        pred_dim = args.pred_dim
        num_classes = args.num_classes
        
        models_dict = {"resnet18": models.resnet18, "resnet50": models.resnet50}
        
        self.exp = args.exp
        self.freeze = args.freeze
        self.backbone = models_dict[args.model](pretrained=args.pretrained)
        in_features = self.backbone.fc.in_features    

        # modules
        if args.model == "resnet18" and args.group_norm:
            self.backbone.bn1 = nn.GroupNorm(
                self.backbone.bn1.num_features // 4, 
                self.backbone.bn1.num_features
            )
            layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
            for i in range(len(layers)):
                if i != 0:
                    layers[i][0].downsample[1] = nn.GroupNorm(
                        layers[i][0].downsample[1].num_features // 4, 
                        layers[i][0].downsample[1].num_features
                    )
                layers[i][0].bn1 = nn.GroupNorm(
                    layers[i][0].bn1.num_features // 4, 
                    layers[i][0].bn1.num_features
                )
                layers[i][0].bn2 = nn.GroupNorm(
                    layers[i][0].bn2.num_features // 4, 
                    layers[i][0].bn2.num_features
                )
                layers[i][1].bn1 = nn.GroupNorm(
                    layers[i][1].bn1.num_features // 4, 
                    layers[i][1].bn1.num_features
                )
                layers[i][1].bn2 = nn.GroupNorm(
                    layers[i][1].bn2.num_features // 4, 
                    layers[i][1].bn2.num_features
                )             

        if self.exp == "simclr":
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, out_dim, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, out_dim, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, out_dim, bias=True)
            )
            
            self.predictor = nn.Linear(out_dim, num_classes, bias=True)

        elif self.exp == "simsiam":
            assert pred_dim != None
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim), 
                nn.ReLU(), 
                nn.Linear(out_dim, in_features, bias=False), 
                nn.BatchNorm1d(in_features), 
                nn.ReLU(), 
                nn.Linear(in_features, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, affine=False)
            )

            self.projector = nn.Sequential(
                nn.Linear(out_dim, pred_dim, bias=False), 
                nn.BatchNorm1d(pred_dim), 
                nn.ReLU(), 
                nn.Linear(pred_dim, out_dim)
            )
            
            self.predictor = nn.Linear(out_dim, num_classes, bias=True)

        # ADDED
        # Loss is compared between dominant positive pairs  
        elif self.exp == "flcl":
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, out_dim, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, out_dim, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, out_dim, bias=True)
            )
            
            self.predictor = nn.Linear(out_dim, num_classes, bias=True)


        elif self.exp in ["FLSL", "centralized"]:
            self.backbone.fc = nn.Identity()
            
            self.predictor = nn.Linear(in_features, num_classes, bias=True)

        elif self.exp == "fedmatch":
            pass

        elif self.exp == "fedorth":
            pass
            
    def set_mode(self, mode):
        self.mode = mode
        if mode == "linear":
            bn_momentum = 0.1
            if self.freeze:
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False

                if self.exp == "simsiam":
                    self.projector.eval()
                    for param in self.projector.parameters():
                        param.requires_grad = False

                                   
            else:
                self.backbone.train()
                for param in self.backbone.parameters():
                    param.requires_grad = True

                if self.exp == "simsiam":
                    self.projector.train()
                    for param in self.projector.parameters():
                        param.requires_grad = True
                    
                    
            
        # For train, unfreeze backbone (and projector for simsiam)
        elif mode == "train":
            bn_momentum = self.args.bn_stat_momentum
            self.backbone.train()
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            
            if self.exp == "simsiam":
                self.projector.train()
                for param in self.projector.parameters():
                    param.requires_grad = True
        
        self.backbone.bn1.momentum = bn_momentum 
        layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        for i in range(len(layers)):
            if i != 0:
                layers[i][0].downsample[1].momentum = bn_momentum 
            layers[i][0].bn1.momentum = bn_momentum 
            layers[i][0].bn2.momentum = bn_momentum 
            layers[i][1].bn1.momentum = bn_momentum 
            layers[i][1].bn2.momentum = bn_momentum 
        

        # Predictor should be always trainable
        self.predictor.train()
        for param in self.predictor.parameters():
            param.requires_grad = True
        
        

            
    def forward(self, x1, x2 = None):
        if self.mode == "linear": 
            if self.freeze:
                with torch.no_grad():
                    z1 = self.backbone(x1)
                return self.predictor(z1)
            
            else:
                return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":           
            if self.exp == "simclr" or self.exp == "flcl":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                assert x2 != None
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.projector(z1)
                p2 = self.projector(z2)
        
                return p1, p2, z1.detach(), z2.detach()

            elif self.exp in ["FLSL", "centralized"]:
                return self.predictor(self.backbone(x1))
