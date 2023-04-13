## Import libraries
import numpy as np
import pandas as pd 
import json
import os
import cv2 as cv
from skimage import morphology
from matplotlib import pyplot as plt
import random

data_dir = './data/' #define data directory
video_dir = './data/videos/' #define videos directory
skeletonized_video_dir = './data/skeletonized_videos/' #define skeletonized videos directory
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

# Function to skeletonize the sign language videos
def skeletonize_video(video_id):
    vid_file_path = f'{video_dir}/{video_id}.mp4'

    cap = cv.VideoCapture(vid_file_path)

    # Initialize the kernel for morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

    # Create an empty list to store the binary images
    binary_image_list = []

    # Extract binary images for each frame of the video
    while True:
        res, frame = cap.read()
        if res:
            # Convert the frame to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Apply Otsu's thresholding method to obtain a binary image
            ret, binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            # Append the binary image to the list
            binary_image_list.append(binary_image)
        else:
            break

    cap.release()

    # Create an empty list to store the skeletonized images
    skeletonized_image_list = []

    # Perform skeletonization on each binary image
    for binary_image in binary_image_list:
        # Apply morphological opening operation
        opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)

        # Threshold the opening image to ensure it only contains 0s and 1s
        opening[opening > 0] = 1

        # Perform skeletonization using skimage's skeletonize function
        skeleton = morphology.skeletonize(opening.astype(np.uint8))

        # Append the skeletonized image to the list
        skeletonized_image_list.append(skeleton)

    # Save the skeletonized images as video
    out_file_path = f'{skeletonized_video_dir}/{video_id}_skeletonized.mp4'

    height, width = skeletonized_image_list[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(out_file_path, fourcc, 30.0, (width, height))

    for skeletonized_image in skeletonized_image_list:
        out.write((skeletonized_image * 255).astype(np.uint8))

    out.release()

def main():

    gloss_inst_df = get_json_as_df()

    # demonstrating use of save_frame_jpg function
    eg_vid_id = gloss_inst_df.iloc[0]['video_id']
    save_frame_jpg(eg_vid_id)
    skeletonize_video(eg_vid_id)
main()