#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
from torchvision import models
import torch
import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])

class ResNet_model(nn.Module):
    def __str__(self):
        return f"""
            ResNet 
                out_dim  : {self.args.out_dim}
                pred_dim : {self.args.pred_dim}
                num_classes: {self.args.num_classes}
                model :  {self.args.model}
                pretrained : {self.args.pretrained}
        """
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
            
            
        elif self.exp == "FL":
            self.backbone.fc = nn.Identity()
            
            self.predictor = nn.Linear(in_features, num_classes, bias=True)
            
    def set_mode(self, mode):
        self.mode = mode
        
        # For linear evaluation, freeze or unfreeze backbone parameters 
        if mode == "linear":
            if self.freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()

                if self.exp == "simsiam":
                    for param in self.projector.parameters():
                        param.requires_grad = False
                    
                    self.projector.eval()

            else:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                self.backbone.train()

                if self.exp == "simsiam":
                    for param in self.projector.parameters():
                        param.requires_grad = True
                    
                    self.projector.train()
            
        # For train, unfreeze backbone (and projector for simsiam)
        elif mode == "train":
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            self.backbone.train()
            if self.exp == "simsiam":
                for param in self.projector.parameters():
                    param.requires_grad = True

                self.projector.train()

        
        # Predictor should be always trainable
        for param in self.predictor.parameters():
            param.requires_grad = True
        
        self.predictor.train()

            
    def forward(self, x1, x2 = None):
        if self.mode == "linear": 
            if self.freeze:
                with torch.no_grad():
                    z1 = self.backbone(x1)
                return self.predictor(z1)
            
            else:
                return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":           
            if self.exp == "simclr":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                assert x2 != None
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.projector(z1)
                p2 = self.projector(z2)
        
                return p1, p2, z1.detach(), z2.detach()
        
            elif self.exp == "FL":
                return self.predictor(self.backbone(x1))
