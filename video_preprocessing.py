## Import libraries
import numpy as np
import pandas as pd 
import json
import os

data_dir = './data/' #define data directory
video_dir = './data/videos/' #define videos directory
wlas_df = pd.read_json(data_dir + 'WLASL_v0.3.json') #returns df with cols ['gloss', 'instances'] - i.e. all json instances for a gloss, e.g. 'book'

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
def main():

    gloss_inst_df = get_json_as_df()

main()