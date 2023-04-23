# from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

keypoints_dir = './data/keypoints/'  # define keypoints directory

def generate_split_data(split, label_map, holistic=True):
    y_labels = [] # ground truth labels
    split_vids = [] # list of all (num_frames,num_keypoints) video npy arrays

    # figuring out which keypoint directory to look in
    if holistic:
        mp_dir = 'holistic'
    else:
        mp_dir = 'pose'

    split_dir = f"{keypoints_dir}{mp_dir}/{split}/"

    # iterate through all available video npy arrays for this particular split
    for npy_name in os.listdir(split_dir):
        if npy_name != '.DS_Store':
            f = os.path.join(split_dir, npy_name)
            
            # sample 50 consecutive frames from this array, since vids have diff lengths
            vid_kp_npy = np.load(f)

            vid_kp = pd.DataFrame(vid_kp_npy).sample(20).sort_index().reset_index(drop=True) 
            print(f"gen_data for vid {npy_name}")

            # extract gloss from the name of the .npy file
            vid_gloss = npy_name.split('_')[1].split('.')[0]

            print(split, npy_name, vid_gloss, vid_kp.shape)

            split_vids.append(vid_kp)
            y_labels.append(label_map[vid_gloss])

    X = np.array(split_vids)
    y = np.array(y_labels)

    return X, y

def main():
    gloss_inst_df = pd.read_pickle("gloss_inst_df.pkl")

    unique_glosses = gloss_inst_df['gloss'].unique().tolist()
    gloss_label_map = {label:num for num, label in enumerate(unique_glosses)}

    X_train, y_train = generate_split_data('train', gloss_label_map, holistic=True)
    X_test, y_test = generate_split_data('test', gloss_label_map, holistic=True)
    X_val, y_val = generate_split_data('val', gloss_label_map, holistic=True)
    
main()
