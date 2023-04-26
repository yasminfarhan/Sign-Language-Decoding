
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

# my files
import video_preprocessing
import gcn_input_gen
import training
import mp_keypoint_extraction
import models as my_models

models_dir = "./models/"
os.makedirs(models_dir, exist_ok=True)

def main():
    random.seed(10)
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

    # training resnet model
    train_ds = video_preprocessing.VideoDataset(ids=split_dict["train"]["video_ids"], labels=split_dict["train"]["labels"], transform=my_models.get_transformer("train"), label_map=gloss_label_map)
    test_ds = video_preprocessing.VideoDataset(ids=split_dict["val"]["video_ids"], labels=split_dict["val"]["labels"], transform=my_models.get_transformer("test"), label_map=gloss_label_map)

    batch_size = 4
    model_type = "rnn"

    if model_type == "rnn":
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

    else:
        train_dl = DataLoader(train_ds, batch_size= batch_size, 
                            shuffle=True, collate_fn= my_models.collate_fn_r3d_18)
        test_dl = DataLoader(test_ds, batch_size= 2*batch_size, 
                            shuffle=False, collate_fn= my_models.collate_fn_r3d_18) 
        model = torch_models.video.r3d_18(pretrained=True, progress=False).float()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_glosses_to_test)

    opt = optim.Adam(model.parameters(), lr=1e-2)
    params_train={
        "num_epochs": 20,
        "optimizer": opt,
        "loss_func": nn.functional.cross_entropy,
        "train_dl": train_dl,
        "val_dl": test_dl,
        "sanity_check": True,
        "lr_scheduler": ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1),
        "path2weights": f"{models_dir}weights_"+model_type+".pt",
        }
    # model,loss_hist,metric_hist = my_models.train_val(model,params_train)

if __name__=="__main__":
    main()