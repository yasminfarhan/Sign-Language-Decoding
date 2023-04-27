
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
import video_preprocessing
import gcn_input_gen
import training
import mp_keypoint_extraction
import models as my_models
import dgcn_training

models_dir = "./models/"
os.makedirs(models_dir, exist_ok=True)

def main():
    random.seed(10)
    holistic = True
    print("In MAIN main()")

    # determining the glosses we're testing, and making sure we have the right info about them
    num_glosses_to_test = 5 #the number of glosses/classes we'll be testing - there are a total of 2000
    gloss_inst_df = video_preprocessing.get_gloss_inst_df()
    glosses_to_test = random.sample(gloss_inst_df['gloss'].unique().tolist(), num_glosses_to_test)
    gloss_label_map = {label:num for num, label in enumerate(glosses_to_test)}
    split_dict = video_preprocessing.get_split_info_dict(glosses_to_test, gloss_inst_df)

    np.save("glosses_to_test.npy", np.array(glosses_to_test))
    
    print(f"glosses: {glosses_to_test}")
    # generating the input to our models from the relevant videos - keypoints & dataframes
    video_preprocessing.extract_video_frames(split_dict["train"]["video_ids"])
    video_preprocessing.extract_video_frames(split_dict["test"]["video_ids"])
    video_preprocessing.extract_video_frames(split_dict["val"]["video_ids"])
    gcn_input_gen.save_vids_gcn_input(gloss_inst_df, glosses_to_test)

    mp_keypoint_extraction.save_vids_keypoints(gloss_inst_df, glosses_to_test)

    ### ResNetRNN parameter initialization, data preparation
    # training resnet model
    train_ds = video_preprocessing.VideoDataset(ids=split_dict["train"]["video_ids"], labels=split_dict["train"]["labels"], transform=my_models.get_transformer("train"), label_map=gloss_label_map)
    test_ds = video_preprocessing.VideoDataset(ids=split_dict["val"]["video_ids"], labels=split_dict["val"]["labels"], transform=my_models.get_transformer("test"), label_map=gloss_label_map)

    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size= batch_size,
                        shuffle=True, collate_fn= my_models.collate_fn_rnn)
    test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                        shuffle=False, collate_fn= my_models.collate_fn_rnn)  
    params_model={
        "num_classes": num_glosses_to_test,
        "dr_rate": 0.1,
        "pretrained" : True,
        "rnn_num_layers": 1,
        "rnn_hidden_size": 100,}
    model = my_models.Resnt18Rnn(params_model).float()

    opt = optim.Adam(model.parameters(), lr=1e-2)
    params_train={
        "num_epochs": 20,
        "optimizer": opt,
        "loss_func": nn.functional.cross_entropy,
        "train_dl": train_dl,
        "val_dl": test_dl,
        "sanity_check": True,
        "lr_scheduler": ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1),
        "path2weights": f"{models_dir}weights_resnet_rnn.pt",
        "model_type": "rnn",
        }
    model,loss_hist,metric_hist = my_models.train_val(model,params_train)

    ### DGCN parameter initialization, data preparation    
    X_train, y_train = training.generate_split_data('train', gloss_label_map, holistic=holistic, kp=False)
    X_test, y_test = training.generate_split_data('test', gloss_label_map, holistic=holistic, kp=False)
    X_val, y_val = training.generate_split_data('val', gloss_label_map, holistic=holistic, kp=False)    

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    train_dl = DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=4)
    val_dl = DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=4)

    params_train["model_type"] = "dgcn"
    params_train["train_dl"] = train_dl
    params_train["val_dl"] = val_dl

    num_features = X_train[0].shape[1]
    model = dgcn_training.DenseGCN(num_features, num_glosses_to_test).float()
    model,loss_hist,metric_hist = my_models.train_val(model,params_train) 

    ## Meta model - i.e. Stacked ensemble with DGCN and ResNetCNN
    train_ds = video_preprocessing.VideoDataset(ids=split_dict["train"]["video_ids"], labels=split_dict["train"]["labels"], transform=my_models.get_transformer("train"), label_map=gloss_label_map, model="meta", split="train")
    test_ds = video_preprocessing.VideoDataset(ids=split_dict["val"]["video_ids"], labels=split_dict["val"]["labels"], transform=my_models.get_transformer("test"), label_map=gloss_label_map, model="meta", split="val")

    train_dl = DataLoader(train_ds, shuffle=False, batch_size=4)
    val_dl = DataLoader(test_ds, shuffle=False, batch_size=4)

    model = my_models.MetaModel(num_features, num_glosses_to_test, params_model).float()

    params_train["model_type"] = "meta"
    params_train["train_dl"] = train_dl
    params_train["val_dl"] = val_dl

    model,loss_hist,metric_hist = my_models.train_val(model,params_train) 
if __name__=="__main__":
    main()