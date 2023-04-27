# Import libraries
import numpy as np
import pandas as pd 
import json
import os
import cv2 as cv
from skimage import morphology
from matplotlib import pyplot as plt
import random
import mp_keypoint_extraction
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict

data_dir = './data/' #define data directory
video_dir = './data/videos/' #define videos directory
video_frames_dir = './data/videos_frames/'
skeletonized_image_dir = './data/skeletonized_image_dir/' #define skeletonized videos directory
wlas_df = pd.read_json(data_dir + 'WLASL_v0.3.json') #returns df with cols ['gloss', 'instances'] - i.e. all json instances for a gloss, e.g. 'book'
holistic = True

# a class that creates a Video dataset by reading video frames, applying transforms, 
# and returning X, y (or more, if also reading in keypoints) in appropriate shape/format for our PyTorch model
class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform, label_map, model=None, split=None, frames_to_sample=20):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.label_map = label_map
        self.frames_to_sample = frames_to_sample
        self.split = split
        self.model = model
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        path2imgs=sorted(glob.glob(video_frames_dir+self.ids[idx]+"/*.jpg"), key=lambda x: int(x.split('_')[-1].split('.')[-2]))
        path2imgs = random.sample(path2imgs, self.frames_to_sample)
        
        label = self.label_map[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame.convert('RGB'))
        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)

        if self.model == "meta":
            mp_dir = 'holistic' if holistic else 'pose'
            path2keypoints = f'{data_dir}gcn_input/{mp_dir}/{self.split}/node_ft_mats/{self.ids[idx]}_{self.labels[idx]}.npy'
            node_ft_tr = torch.tensor(np.load(path2keypoints))

            return node_ft_tr, frames_tr, torch.tensor(label)
        else:
            # return frames_tr, F.one_hot(torch.tensor(label), len(self.label_map.keys()))
            return frames_tr, torch.tensor(label)
    
# function to return list of included video ids according to what's been downloaded
# for a particular gloss' instance e.g. return all video ids for 'book' gloss
def incl_video_ids(json_list):
    videos_list = []    
    for ins in json_list:
        video_id = ins['video_id']
        file_path = f'{video_dir}/{video_id}.mp4'
        if os.path.exists(file_path):
            videos_list.append(video_id)
    return videos_list

# function to convert our list of json instances per gloss into pandas df where each row is one 
# video instance w/columns from json features 'bbox', 'fps', etc
def get_json_as_df():
    with open(data_dir + 'WLASL_v0.3.json', 'r') as data_file:
        json_data = data_file.read()

    json_gloss_instances = json.loads(json_data)
    json_df_full = pd.DataFrame()

    # generate pd dataframe for all gloss instances, not including instances for which we don't have the video ids
    for i in range(len(json_gloss_instances)):
        # retrieve features for this gloss
        gloss = json_gloss_instances[i]['gloss'] #get gloss label for this list of instances
        gloss_inst_lst = json_gloss_instances[i]['instances']

        # build gloss specific dataframe
        gloss_df = pd.DataFrame.from_dict(gloss_inst_lst) #convert list of instances for this gloss to pd DataFrame where each row is dedicated to 1 instance
        gloss_df = gloss_df.loc[gloss_df['video_id'].isin(incl_video_ids(gloss_inst_lst))] #exclude instances where video id is not in included vids list
        gloss_df.insert(0, 'gloss', gloss) #insert 'gloss' column at position 0 to denote the gloss label across all these new rows

        # concatenate this gloss' dataframe to the df containing rows for all gloss' instances
        json_df_full = pd.concat([json_df_full, gloss_df], ignore_index=True) 

    return json_df_full

# function which extracts one random frame from a video, and saves to current dir as .jpg
def save_frame_jpg(video_id):
    vid_file_path = f'{video_dir}/{video_id}.mp4'

    cap = cv.VideoCapture(vid_file_path)

    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    frame_number = random.randint(0, totalFrames)

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)
    res, frame = cap.read()

    if res:
        #Set grayscale colorspace for the frame. 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #Cut the video extension to have the name of the video
        my_video_name = video_id.split(".")[0]

        # #Display the resulting frame
        # cv.imshow(my_video_name+' frame '+ str(frame_number),gray)

        # #Set waitKey - displays image for this duration in ms
        # cv.waitKey(2000)

        #Store this frame to an image
        cv.imwrite(my_video_name+'_frame_'+str(frame_number)+'.jpg',gray)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()
    else:
        print("ERROR: save_frame_jpg() - failed to retrieve frame from video with ID {}".format(video_id))


# function which extracts all frames of a video, and saves to subdir with corresponding subfolders for video ID
def extract_video_frames(video_id_lst, skeletonize=False):
    for video_id in video_id_lst:
        vid_file_path = f'{video_dir}/{video_id}.mp4'

        # create subfolder for video ID
        sub_dir = skeletonized_image_dir if skeletonize else video_frames_dir
        subfolder_path = f"{sub_dir}/{video_id}"

        if not(os.path.exists(subfolder_path)):
            os.makedirs(subfolder_path, exist_ok=True)
            cap = cv.VideoCapture(vid_file_path)

            totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            frame_number = 0

            while frame_number < totalFrames:
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    break

                if skeletonize:
                    # Set grayscale colorspace for the frame. 
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    # Perform skeletonization
                    frame = cv.ximgproc.thinning(gray, None, cv.ximgproc.THINNING_ZHANGSUEN)

                # Store the skeletonized image to a file in subfolder for video ID
                frame_path = f'{subfolder_path}/{video_id}_frame_{frame_number}.jpg'

                # cv.imwrite(frame_path, skeleton)
                cv.imwrite(frame_path, frame)

                frame_number += 1

            # When everything done, release the capture
            cap.release()
            cv.destroyAllWindows()

# create a gloss_inst dataframe containing the information about all our video 
# instances if it doesn't exist, otherwise just retrieve it from the saved pkl file
def get_gloss_inst_df():
    if os.path.exists("gloss_inst_df.pkl"):
        return pd.read_pickle("gloss_inst_df.pkl")
    else:
        gloss_inst_df = get_json_as_df()
        gloss_inst_df.to_pickle("gloss_inst_df.pkl")
        return gloss_inst_df

# retrieve the video ids / splits for all glosses we're testing
def get_split_info_dict(gloss_lst, gloss_df):

    # create a defaultdict that initializes new inner dictionaries to an empty list
    attr_map = lambda: defaultdict(list)
    split_map = defaultdict(attr_map)

    gloss_groups = gloss_df.groupby('gloss') #group rows in json df with all instances by gloss    

    for gloss, gloss_df in gloss_groups:
        if gloss in gloss_lst:
            split_groups = gloss_df.groupby('split') #group rows in the df for this gloss by type of split; not all glosses have every type

            for split, split_df in split_groups: 
                split_df = split_df.reset_index(drop=True)    

                for i in range(split_df.shape[0]):
                    video_id = split_df.loc[i, 'video_id']
                    split_map[split]["labels"].append(gloss)
                    split_map[split]["video_ids"].append(video_id)
    return split_map

def main():
    pass
if __name__=="__main__":
    main()