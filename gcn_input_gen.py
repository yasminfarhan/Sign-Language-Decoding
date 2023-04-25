import numpy as np
import pandas as pd
from mp_keypoint_extraction import extract_vid_keypoints
from video_preprocessing import get_json_as_df
import os

gcn_input_dir = './data/gcn_input/'  # define gcn input directory
os.makedirs(gcn_input_dir, exist_ok=True)

# generate adjacency and feature matrices for all videos across some subset of glosses,
# across splits, then save each as numpy arrays in dedicated directory
def save_vids_gcn_input(gloss_inst_df, holistic=True):
    gloss_subset_num = 5 # - the number of glosses we are testing - get all train, val, test data for them - TODO - change to 2000 when testing all glosses
    mp_dir = 'holistic' if holistic else 'pose'
    gloss_groups = gloss_inst_df.groupby('gloss') #group rows in json df with all instances by gloss
    gloss_lst = []

    cnt = 0
    for gloss, gloss_df in gloss_groups:

        if cnt == gloss_subset_num:
            break
        else:
            cnt += 1
            gloss_lst.append(gloss)

        split_groups = gloss_df.groupby('split') #group rows in the df for this gloss by type of split; not all glosses have every type

        for split, split_df in split_groups:
            split_df = split_df.reset_index(drop=True)
            adj_file_path = f"{gcn_input_dir}{mp_dir}/{split}/adj_mats/" #specify path to our adjacency matrices (NxN)
            node_ft_file_path = f"{gcn_input_dir}{mp_dir}/{split}/node_ft_mats/" #specify path to our node feature matrices (NxD)

            os.makedirs(adj_file_path, exist_ok=True)
            os.makedirs(node_ft_file_path, exist_ok=True)

            print(f"Saving gcn_input for gloss {gloss}, split {split}")
            for i in range(split_df.shape[0]):
                vid_info = split_df.loc[i, ['video_id']]
                vid_id = vid_info['video_id']

                # generate keypoint features from keypoints, then adjacency matrix from keypoint features
                kp_ft = create_keypoint_features(vid_id)
                kp_adj = create_adjacency_matrix(kp_ft)

                # save keypoint features, adjacency matrix for this video 
                np.save(f"{node_ft_file_path}{vid_id}_{gloss}.npy", np.array(kp_ft))
                np.save(f"{adj_file_path}{vid_id}_{gloss}.npy", np.array(kp_adj))
    np.save("glosses_to_test_gcn.npy", np.array(gloss_lst)) #save the glosses to a numpy array so that training.py can read them

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
    print("In gcn_input_gen main()")

    # Load the JSON file and get the DataFrame
    gloss_inst_df = get_json_as_df()

    # # Extract the video_id from the DataFrame
    # video_id = gloss_inst_df.loc[0, 'video_id']

    # # Create the keypoint features for the given video_id
    # keypoint_features = create_keypoint_features(video_id)

    # # Create the adjacency matrix for the keypoints
    # adjacency_matrix = create_adjacency_matrix(keypoint_features)

    # # Save the keypoint_features and adjacency_matrix as .npy files
    # np.save('keypoint_features.npy', keypoint_features)
    # np.save('adjacency_matrix.npy', adjacency_matrix)

    gloss_inst_df.to_pickle("gloss_inst_df.pkl")
    gloss_inst_df = pd.read_pickle("gloss_inst_df.pkl")

    # generating input to our GCN
    save_vids_gcn_input(gloss_inst_df)

if __name__=="__main__":
    main()