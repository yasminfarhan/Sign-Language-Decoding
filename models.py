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
import dgcn_training as dgcn

class MetaModel(nn.Module):
  def __init__(self, num_features, num_classes, params_model):
    super(MetaModel, self).__init__()
    self.model1 = dgcn.DenseGCN(num_features, num_classes).float() #expects batch,nodes,ft
    self.model2 = Resnt18Rnn(params_model).float()
    self.params_model = params_model

  def forward(self, x1, a1, x2):
    out1 = self.model1(x1,a1)
    out2 = self.model2(x2)
    return (out1 + out2)/2

# TODO - link-to-source
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

def collate_fn_rnn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l).clone() for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

def get_transformer(split):
    h, w =224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.RandomHorizontalFlip(p=0.5),  
                transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ]) 
    test_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ]) 
    
    if split == "train":
        return train_transformer
    else:
        return test_transformer
    
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_epoch_rnn(model,loss_func,dataset_dl,sanity_check=False,opt=None,model_type=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in tqdm_notebook(dataset_dl):

        # xb=xb.to(device)
        # yb=yb.to(device)

        output=model(xb)

        print("model:", model_type)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric


def loss_epoch_dgcn(model,loss_func,dataset_dl,sanity_check=False,opt=None,model_type=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)

    print(model_type)

    for xb, yb in tqdm_notebook(dataset_dl):

        # xb=xb.to(device)
        # yb=yb.to(device)

        #fully connected adjacency matrix w/diagonals set to 0 for later self-loop addition
        adj = torch.ones(xb.shape[1],xb.shape[1]).fill_diagonal_(0)
        output=model(xb.float(), adj)

        print("model:", model_type)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric


def loss_epoch_meta(model,loss_func,dataset_dl,sanity_check=False,opt=None,model_type=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)

    print(model_type)

    for x_dgcn, x_rnn, yb in tqdm_notebook(dataset_dl):
        # xb=xb.to(device)
        # yb=yb.to(device)

        #fully connected adjacency matrix w/diagonals set to 0 for later self-loop addition
        adj = torch.ones(x_dgcn.shape[1],x_dgcn.shape[1]).fill_diagonal_(0)
        output=model(x_dgcn.float(), adj, x_rnn.float())     

        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break

    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

def loss_batch(loss_func, output, target, opt=None):
    # print(output.dtype, target.dtype)
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    model_type=params["model_type"]
    
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        if(model_type == "rnn"):
            train_loss, train_metric=loss_epoch_rnn(model,loss_func,train_dl,sanity_check,opt,model_type)
        elif(model_type == "dgcn"):
            train_loss, train_metric=loss_epoch_dgcn(model,loss_func,train_dl,sanity_check,opt,model_type)
        elif(model_type == "meta"):
            train_loss, train_metric=loss_epoch_meta(model,loss_func,train_dl,sanity_check,opt,model_type)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad(): #only call no grad for validation - because we don't care abt backprop
            if(model_type == "rnn"):
                val_loss, val_metric=loss_epoch_rnn(model,loss_func,val_dl,sanity_check)
            elif(model_type == "dgcn"):
                val_loss, val_metric=loss_epoch_dgcn(model,loss_func,val_dl,sanity_check)
            elif(model_type == "meta"):
                val_loss, val_metric=loss_epoch_meta(model,loss_func,val_dl,sanity_check)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history
def main():
    pass
if __name__=="__main__":
    main()     