import os
from torch.utils.data import Dataset, DataLoader, Subset
import glob
from PIL import Image
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from torch import nn
from torchvision import models
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from torch_geometric.nn.dense import DenseGCNConv
import torch.nn.functional as F

class MetaModel(nn.Module):
  def __init__(self, num_features, num_classes, params_model):
    super(MetaModel, self).__init__()
    self.model1 = DenseGCN(num_features, num_classes).float() #expects batch,nodes,ft
    self.model2 = Resnt18Rnn(params_model).float()
    self.params_model = params_model
    self.lin = nn.Linear(num_classes, num_classes)

  def forward(self, x1, a1, x2):
    out1 = self.model1(x1,a1)
    out2 = self.model2(x2)
    y_meta = self.lin(out1+out2)
    return y_meta

class DenseGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=8):
        super(DenseGCN, self).__init__()
        self.conv1 = DenseGCNConv(num_features, num_hidden)
        self.conv2 = DenseGCNConv(num_hidden, num_hidden)
        self.conv3 = DenseGCNConv(num_hidden, num_hidden)
        self.fc = torch.nn.Linear(num_hidden, num_classes)

    def forward(self, x, a):
        x = self.conv1(x, a)
        x = F.tanh(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, a)
        x = F.tanh(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, a)
        x = F.tanh(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.mean(x, dim=1)

        x = self.fc(x)
        return x

# Model definition from https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/Chapter10.ipynb
class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
    def forward(self, x):
        batch_size, timesteps, channels, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, timesteps):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    