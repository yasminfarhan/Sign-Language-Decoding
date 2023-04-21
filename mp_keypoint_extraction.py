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
    
def extract_frame_keypoints(frame_results, holistic=True):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in frame_results.pose_landmarks.landmark]).flatten() if frame_results.pose_landmarks else np.zeros(132)

    if holistic:
        face = np.array([[res.x, res.y, res.z] for res in frame_results.face_landmarks.landmark]).flatten() if frame_results.face_landmarks else np.zeros(1404)
        lh = np.array([[res.x, res.y, res.z] for res in frame_results.left_hand_landmarks.landmark]).flatten() if frame_results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in frame_results.right_hand_landmarks.landmark]).flatten() if frame_results.right_hand_landmarks else np.zeros(21*3)

        return np.concatenate([pose, face, lh, rh])
    else:
        return pose

# extracts keypoints for every frame in the video and saves to .npy array file of shape (num_frames, 1662)
def extract_vid_keypoints(video_id, holistic=True, display=False):

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

            frame_keypoints = extract_frame_keypoints(results)

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

    # Save keypoints_list for the current video_id
    if holistic:
        path_suffix = '_holistic'
    else:
        path_suffix = ''

    keypoints_file_path = f"{keypoints_dir}/{video_id}{path_suffix}.npy"
    np.save(keypoints_file_path, frames_keypoints_lst)

    return results

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