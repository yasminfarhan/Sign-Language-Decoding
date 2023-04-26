import numpy as np
import pandas as pd
from mp_keypoint_extraction import extract_vid_keypoints
from video_preprocessing import get_json_as_df
import os

gcn_input_dir = './data/gcn_input/'  # define gcn input directory
os.makedirs(gcn_input_dir, exist_ok=True)

# generate adjacency and feature matrices for all videos across some subset of glosses,
# across splits, then save each as numpy arrays in dedicated directory
def save_vids_gcn_input(gloss_inst_df, glosses_to_test, holistic=True):
    mp_dir = 'holistic' if holistic else 'pose'
    gloss_groups = gloss_inst_df.groupby('gloss') #group rows in json df with all instances by gloss

    for gloss, gloss_df in gloss_groups:

        if gloss in glosses_to_test:
            split_groups = gloss_df.groupby('split') #group rows in the df for this gloss by type of split; not all glosses have every type

            for split, split_df in split_groups:
                split_df = split_df.reset_index(drop=True)
                adj_file_path = f"{gcn_input_dir}{mp_dir}/{split}/adj_mats/" #specify path to our adjacency matrices (NxN)
                node_ft_file_path = f"{gcn_input_dir}{mp_dir}/{split}/node_ft_mats/" #specify path to our node feature matrices (NxD)

                os.makedirs(adj_file_path, exist_ok=True)
                os.makedirs(node_ft_file_path, exist_ok=True)

                print(f"Saving gcn_input for gloss {gloss}, split {split}")
                for i in range(split_df.shape[0]):
                    vid_id = split_df.loc[i, 'video_id']
                    
                    ft_path = f"{node_ft_file_path}{vid_id}_{gloss}.npy"
                    adj_path = f"{adj_file_path}{vid_id}_{gloss}.npy"

                    # save keypoint features, adjacency matrix for this video 
                    if not(os.path.exists(ft_path)):
                        # generate keypoint features from keypoints
                        kp_ft = create_keypoint_features(vid_id)
                        np.save(ft_path, np.array(kp_ft))
                    
                    if not(os.path.exists(adj_path)): #don't bother generating again
                        # generate adjacency matrix from keypoint features
                        kp_adj = create_adjacency_matrix(kp_ft)
                        np.save(adj_path, np.array(kp_adj))

# extract the keypoint features, and reshape the mean keypoint values across frames to obtain matrix of size N, D
# where N is the number of nodes and D is the number of input features (x, y, z, visibility)
def create_keypoint_features(video_id):
    # Extract keypoints for the video_id using the extract_vid_keypoints function
    keypoints_list, _ = extract_vid_keypoints(video_id, gcn_input=True)

    # Convert keypoints list into a NumPy array
    keypoints_array = np.array(keypoints_list)

    # Calculate the mean keypoints across all frames of the video
    mean_keypoints = np.mean(keypoints_array, axis=0)

    return mean_keypoints.reshape(-1, 4)

# create adjacency matrix of size N,N where N is the number of nodes in the graph
def create_adjacency_matrix(keypoint_features):
    num_keypoints = keypoint_features.shape[0]
    adjacency_matrix = np.zeros((num_keypoints, num_keypoints))

    for i in range(num_keypoints):
        for j in range(num_keypoints):
            distance = np.linalg.norm(keypoint_features[i, :2] - keypoint_features[j, :2])
            adjacency_matrix[i, j] = 1 / (1 + distance)

    return adjacency_matrix

def main():
    pass
if __name__=="__main__":
    main()