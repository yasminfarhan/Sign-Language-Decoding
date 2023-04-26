import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.linalg import fractional_matrix_power
import scipy.sparse as sp
from torch_geometric.nn.dense import DenseGCNConv
import torch.utils.data as data
import gcn_input_gen
import pandas as pd
from training import generate_split_data

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


def load_data_dense(data_dir, split, gloss_label_map):
    node_ft_dir = os.path.join(data_dir, "holistic", split, "node_ft_mats")
    adj_dir = os.path.join(data_dir, "holistic", split, "adj_mats")
    
    data_list = []
    label_list = []
    for file in os.listdir(node_ft_dir):
        if not file.endswith(".npy"):
            continue
        node_ft_path = os.path.join(node_ft_dir, file)
        adj_path = os.path.join(adj_dir, file)

        x = torch.tensor(np.load(node_ft_path), dtype=torch.float)#.t()
        adj_matrix = torch.tensor(np.load(adj_path), dtype=torch.float)
        
        gloss = file.split("_")[-1].split(".")[0]
        label = torch.tensor(gloss_label_map[gloss])

        data_list.append((x, label))
    
    return data_list

# function to train the DGCN model
def train_model(model, train_data, val_data, epochs=2000, lr=0.001):
    train_dl = data.DataLoader(train_data, shuffle=False, batch_size=4)
    val_dl = data.DataLoader(val_data, shuffle=False, batch_size=4)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y in train_dl:
            y_pred = model(x.float(), torch.ones(x.shape[1],x.shape[1]).fill_diagonal_(0))
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y.float())
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))

# function to validate DGCN model during training
def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y in valid_dl:
        y_hat = model(x.float(), torch.ones(x.shape[1],x.shape[1]).fill_diagonal_(0)) #filling diag w/0 assuming dense gcnconv adds self loops internally
        loss = F.cross_entropy(y_hat, y.float())
        pred = torch.max(y_hat, 1)[1]
        yt = torch.max(y, 1)[1]
        correct += (pred == yt).float().sum()
        total += yt.shape[0]
        sum_loss += loss.item()*yt.shape[0]
    return sum_loss/total, correct/total

def main():    
    holistic = True
    
    glosses_to_test = np.load("glosses_to_test.npy")
    print(glosses_to_test)

    gloss_label_map = {label: num for num, label in enumerate(glosses_to_test)}
    num_classes = len(glosses_to_test)

    X_train, y_train = generate_split_data('train', gloss_label_map, holistic=holistic, kp=False)
    X_test, y_test = generate_split_data('test', gloss_label_map, holistic=holistic, kp=False)
    X_val, y_val = generate_split_data('val', gloss_label_map, holistic=holistic, kp=False)

    # transforming labels to one hot vector representation
    y_train_ohe = F.one_hot(torch.tensor(y_train), num_classes)
    y_test_ohe = F.one_hot(torch.tensor(y_test), num_classes)
    y_val_ohe = F.one_hot(torch.tensor(y_val), num_classes)

    num_features = X_train[0].shape[1]
    model = DenseGCN(num_features, num_classes).float()
    train_model(model, data.TensorDataset(torch.tensor(X_train), y_train_ohe), data.TensorDataset(torch.tensor(X_val), y_val_ohe))

if __name__ == "__main__":
    main()
