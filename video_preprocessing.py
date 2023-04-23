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

data_dir = './data/' #define data directory
video_dir = './data/videos/' #define videos directory
skeletonized_image_dir = './data/skeletonized_image_dir/' #define skeletonized videos directory
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


# function which skeletonizes all frames of a video, and saves to skeletonized_image_dir with corresponding subfolders for video ID
def skeletonize_video_frames(video_id):
    vid_file_path = f'{video_dir}/{video_id}.mp4'

    # create subfolder for video ID
    subfolder_path = f"{skeletonized_image_dir}/{video_id.split('.')[0]}"
    os.makedirs(subfolder_path, exist_ok=True)

    cap = cv.VideoCapture(vid_file_path)

    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    frame_number = 0

    while frame_number < totalFrames:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        res, frame = cap.read()

        if res:
            # Set grayscale colorspace for the frame. 
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Perform skeletonization
            skeleton = cv.ximgproc.thinning(gray, None, cv.ximgproc.THINNING_ZHANGSUEN)

            # Store the skeletonized image to a file in subfolder for video ID
            frame_path = f'{subfolder_path}/{video_id.split(".")[0]}_frame_{frame_number}.jpg'
            cv.imwrite(frame_path, skeleton)

            frame_number += 1

        else:
            print(f"ERROR: skeletonize_video_frames() - failed to retrieve frame {frame_number} from video with ID {video_id}")
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

def main():

    gloss_inst_df = get_json_as_df()
    gloss_inst_df.to_pickle("gloss_inst_df.pkl")

    gloss_inst_df = pd.read_pickle("gloss_inst_df.pkl")

    mp_keypoint_extraction.save_vids_keypoints(gloss_inst_df)

main()