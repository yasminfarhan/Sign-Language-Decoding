import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd 

video_dir = './data/videos/' #define videos directory
keypoints_dir = './data/keypoints/'  # define keypoints directory
os.makedirs(keypoints_dir, exist_ok=True)

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

## returns image, and unprocessed keypoint results for some frame in the video
def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
    
def extract_frame_keypoints(frame_results, holistic=True, gcn_input=False):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in frame_results.pose_landmarks.landmark]).flatten() if frame_results.pose_landmarks else np.zeros(132)

    if holistic:
        if gcn_input: #appending face, lh, rh input with 0 to retain pose visibility value
            face = np.array([[res.x, res.y, res.z, 0] for res in frame_results.face_landmarks.landmark]).flatten() if frame_results.face_landmarks else np.zeros(468*4)
            lh = np.array([[res.x, res.y, res.z, 0] for res in frame_results.left_hand_landmarks.landmark]).flatten() if frame_results.left_hand_landmarks else np.zeros(21*4)
            rh = np.array([[res.x, res.y, res.z, 0] for res in frame_results.right_hand_landmarks.landmark]).flatten() if frame_results.right_hand_landmarks else np.zeros(21*4)
        else:
            face = np.array([[res.x, res.y, res.z] for res in frame_results.face_landmarks.landmark]).flatten() if frame_results.face_landmarks else np.zeros(1404)
            lh = np.array([[res.x, res.y, res.z] for res in frame_results.left_hand_landmarks.landmark]).flatten() if frame_results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in frame_results.right_hand_landmarks.landmark]).flatten() if frame_results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    else:
        return pose

# extracts keypoints for every frame in the video and saves to .npy array file of shape (num_frames, 1662)
def extract_vid_keypoints(video_id, holistic=True, display=False, gcn_input=False):

    # Set mediapipe model 
    vid_file_path = f'{video_dir}/{video_id}.mp4'
    cap = cv.VideoCapture(vid_file_path)
    frames_keypoints_lst = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            frame_keypoints = extract_frame_keypoints(results, gcn_input=gcn_input)

            if display:
                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Show to screen
                cv.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

            frames_keypoints_lst.append(frame_keypoints)

        cap.release()
        cv.destroyAllWindows()    

    return frames_keypoints_lst, results

# iterates through available videos for each kind of split and saves as npy array
def save_vids_keypoints(gloss_inst_df, holistic=True):
    gloss_subset_num = 20 # - the number of glosses we are testing - get all train, val, test data for them - TODO - change to 2000 when testing all glosses
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
            split_file_path = f"{keypoints_dir}{mp_dir}/{split}/"
            os.makedirs(split_file_path, exist_ok=True)

            print(f"Saving keypoint npys for gloss {gloss}, split {split}")
            for i in range(split_df.shape[0]):
                vid_info = split_df.loc[i, ['video_id']]
                vid_id = vid_info['video_id']

                frames_keypoints_lst, _ = extract_vid_keypoints(vid_id)

                # save keypoints for this video
                vid_file_path = f"{split_file_path}{vid_id}_{gloss}.npy"
                np.save(vid_file_path, np.array(frames_keypoints_lst))
    np.save("glosses_to_test.npy", np.array(gloss_lst)) #save the glosses to a numpy array so that training.py can read them

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 