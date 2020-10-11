import torch
from torch import nn as nn
from models.ResNet import *
from loss.loss import *
class Conv3D(nn.Module):
    def __init__(self, num_classes=10):
        super(Conv3D, self).__init__()
        self.conv_layers = resnet18()
        self.s=64.0
        self.m=1.25
        self.loss=LMCL(512,10,s=self.s,m=self.m)
        self.loss_base=nn.CrossEntropyLoss()
        self.fc_=nn.Linear(512,10)
    def forward(self, x, labels=None,test=False):
        x= self.conv_layers(x)
        if test:
            return x
        # assert labels is not None : "Labels aren't fed to model"
        # print(x.shape)
        loss_a=self.loss_base(self.fc_(x),labels)
        loss_b=self.loss(x,labels)
        loss=loss_a+loss_b*1e-1
        # print(float(loss_a),float(loss_b))
        return {'conv':x,'loss':loss}
