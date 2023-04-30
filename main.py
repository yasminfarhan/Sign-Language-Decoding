
import os
from torch.utils.data import Dataset, DataLoader, Subset
import glob
from PIL import Image
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from torch import nn
from torchvision import models as torch_models
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.utils.data as data
# from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

# my files
import data_utils
import dgcn_utils
import kp_utils
import training_utils
import training
import models as my_models


models_dir = "./models/"
os.makedirs(models_dir, exist_ok=True)

# kicks off generation of input data to the DGCN, ResNet models and returns an enumerated
# representation of the glosses and a split_dict that allows for labels, video_ids per 
# training split to be extracted by the VideoDataset object with ease
def gen_input_data(glosses_to_test, gloss_inst_df):
    split_dict = data_utils.get_split_info_dict(glosses_to_test, gloss_inst_df)

    np.save("glosses_to_test.npy", np.array(glosses_to_test))
    
    print(f"glosses: {glosses_to_test}")

    # generating the input to our models from the relevant videos - keypoints & dataframes
    data_utils.extract_video_frames(split_dict["train"]["video_ids"])
    data_utils.extract_video_frames(split_dict["test"]["video_ids"])
    data_utils.extract_video_frames(split_dict["val"]["video_ids"])
    dgcn_utils.save_vids_gcn_input(gloss_inst_df, glosses_to_test)

    kp_utils.save_vids_keypoints(gloss_inst_df, glosses_to_test)

    gloss_label_map = {label:num for num, label in enumerate(glosses_to_test)}

    return split_dict, gloss_label_map

def main():
    random.seed(15)
    holistic = True

    # determining the glosses we're testing, and making sure we have the right info about them
    num_glosses_to_test = 1 #the number of glosses/classes we'll be testing - there are a total of 2000
    gloss_inst_df = data_utils.get_gloss_inst_df()
    gloss_counts = gloss_inst_df.groupby('split').get_group('train')['gloss'].value_counts()
    glosses_to_test = gloss_counts[:num_glosses_to_test].index.tolist() #get the glosses with the most training examples
    split_dict, gloss_label_map = gen_input_data(glosses_to_test, gloss_inst_df)

    ### ResNetRNN parameter initialization, data preparation
    # training resnet model
    train_ds = data_utils.VideoDataset(ids=split_dict["train"]["video_ids"], labels=split_dict["train"]["labels"], transform=data_utils.get_transformer("train"), label_map=gloss_label_map)
    test_ds = data_utils.VideoDataset(ids=split_dict["val"]["video_ids"], labels=split_dict["val"]["labels"], transform=data_utils.get_transformer("test"), label_map=gloss_label_map)

    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size= batch_size,
                        shuffle=True, collate_fn= data_utils.collate_fn_rnn)
    test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                        shuffle=True, collate_fn= data_utils.collate_fn_rnn)  
    params_model={
        "num_classes": num_glosses_to_test,
        "dr_rate": 0.1,
        "pretrained" : True,
        "rnn_num_layers": 1,
        "rnn_hidden_size": 100,}
    model = my_models.Resnt18Rnn(params_model).float()

    opt = optim.Adam(model.parameters(), lr=1e-2)
    params_train={
        "num_epochs": 10,
        "optimizer": opt,
        "loss_func": nn.functional.cross_entropy,
        "train_dl": train_dl,
        "val_dl": test_dl,
        "sanity_check": True,
        "lr_scheduler": ReduceLROnPlateau(opt, mode='min',factor=0.8, patience=5,verbose=1),
        "path2weights": f"{models_dir}weights_resnet_rnn.pt",
        "model_type": "rnn",
        }
    # model,loss_hist,metric_hist = training_utils.train_val(model,params_train)

    ### DGCN parameter initialization, data preparation    
    X_train, y_train = training.generate_split_data('train', gloss_label_map, holistic=holistic, kp=False)
    X_test, y_test = training.generate_split_data('test', gloss_label_map, holistic=holistic, kp=False)
    X_val, y_val = training.generate_split_data('val', gloss_label_map, holistic=holistic, kp=False)    

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    train_dl = DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=4)
    val_dl = DataLoader(data.TensorDataset(X_val, y_val), shuffle=True, batch_size=4)

    params_train["model_type"] = "dgcn"
    params_train["train_dl"] = train_dl
    params_train["val_dl"] = val_dl

    num_features = X_train[0].shape[1]
    model = my_models.DenseGCN(num_features, num_glosses_to_test).float()
    # model,loss_hist,metric_hist = training_utils.train_val(model,params_train) 

    ## Meta model - i.e. Stacked ensemble with DGCN and ResNetCNN
    train_ds = data_utils.VideoDataset(ids=split_dict["train"]["video_ids"], labels=split_dict["train"]["labels"], transform=data_utils.get_transformer("train"), label_map=gloss_label_map, model="meta", split="train")
    test_ds = data_utils.VideoDataset(ids=split_dict["val"]["video_ids"], labels=split_dict["val"]["labels"], transform=data_utils.get_transformer("test"), label_map=gloss_label_map, model="meta", split="val")

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=4)
    val_dl = DataLoader(test_ds, shuffle=True, batch_size=4)

    params_train["model_type"] = "meta"
    params_train["train_dl"] = train_dl
    params_train["val_dl"] = val_dl

    epochs = [10, 50, 100, 200]
    lrates = [1e-1, 1e-2, 1e-3, 1e-4]
    batch_size = [4, 8, 12]
    num_g_to_test = [5, 10, 15, 20]

    ### Hyperparameter tuning for Meta Model
    num_iters = 1
    for i in range(num_iters):
        bs = random.choice(batch_size)
        lr = random.choice(lrates)
        ep = random.choice(epochs)
        num_g = random.choice(num_g_to_test)

        print(f"TESTING: meta, batch size: {bs} LR: {lr} Epochs: {ep} top glosses: {num_g}")

        glosses = gloss_counts[:num_g].index.tolist()
        split_dict, gloss_label_map = gen_input_data(glosses, gloss_inst_df)

        model = my_models.MetaModel(num_features, num_g, params_model).float()

        train_ds_n = data_utils.VideoDataset(ids=split_dict["train"]["video_ids"], labels=split_dict["train"]["labels"], transform=data_utils.get_transformer("train"), label_map=gloss_label_map, model="meta", split="train")
        test_ds_n = data_utils.VideoDataset(ids=split_dict["val"]["video_ids"], labels=split_dict["val"]["labels"], transform=data_utils.get_transformer("test"), label_map=gloss_label_map, model="meta", split="val")

        train_dl = DataLoader(train_ds_n, shuffle=True, batch_size=bs)
        val_dl = DataLoader(test_ds_n, shuffle=True, batch_size=bs)

        opt = optim.Adam(model.parameters(), lr=lr)
        sch = random.choice([ReduceLROnPlateau(opt, mode='min',factor=0.1, patience=5,verbose=1)])
        params_train["lr_scheduler"] = sch
        params_train["num_epochs"] = ep
        params_train["optimizer"] = opt

        model,loss_hist,metric_hist = training_utils.train_val(model,params_train)

        np.save(f"meta_loss_{bs}_{lr}_{ep}_{num_g}.npy", loss_hist)
        np.save(f"meta_acc_{bs}_{lr}_{ep}_{num_g}.npy", metric_hist)

        # print(loss_hist)
        plt.plot(loss_hist["train"]) 
        plt.plot(loss_hist["val"])
        plt.title("Training, Validation Loss for best performing hyperparameters")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.show()

        plt.plot(metric_hist["train"]) 
        plt.plot(metric_hist["val"])
        plt.title("Training, Validation Accuracy for best performing hyperparameters")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.show()

        print(model(split_dict["test"]["video_ids"]))
if __name__=="__main__":
    main()