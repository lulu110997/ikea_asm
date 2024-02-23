'''
# The landmarker is initialized. Use it here.
# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# The three Kinect V2 cameras are triggered to capture the assembly activities  simultaneously in real
# time (∼24 fps)
# There are in total 1113 RGB videos and 371 depth videos (top view). Overall, the dataset contains 3,046,977
# frames (∼35.27h) of footage with an average of 2735.2 frames per video (∼1.89min).
# The dataset contains a total of 16,764 annotated actions with an average of 150 frames per action (∼6sec)
# TODO: For a full list of action names and ids, see supplemental
# Temporally, we specify the boundaries (start and end frame) of all atomic actions in the video from a pre-defined set.
# TODO: atomic actions?
# We also annotated the human skeleton of the subjects involved assembly... Annotated 12 body joints... Due to
# occlusion with furniture, self-occlusions and uncommon  human poses, we include a confidence value between 1
# and 3 along with the annotation
# The dataset contains 2D human joint annotations in the COCO format [39] for 1% of frames, the same keyframes
# selected for instance segmentation, which cover a diverse range of human poses across each video.
# We also obtain pseudo-ground-truth 3D annotations by fine-tuning a Mask R-CNN [22] 2D joint detector on the
# labeled data, and triangulating the detections of the model from the three calibrated camera views
# https://arxiv.org/pdf/2007.00394.pdf
'''
from typing import Tuple, List, Any
from natsort import natsort
from tqdm import tqdm
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['GLOG_minloglevel'] = '3'
# import logging
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
# logging.getLogger('mediapipe').setLevel(logging.ERROR)
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger('absl').setLevel(logging.ERROR)
import logging
logging.basicConfig(filename='train.log', level=logging.INFO)

import json
import argparse
import cv2
import numpy as np
import mediapipe as mp

PATH_TO_MODEL = '/home/louis/Data/Fernandez_HAR/hand_landmarker.task'
OFFSET = 120  # (y,x)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a hand landmarker instance with image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=PATH_TO_MODEL, delegate=mp.tasks.BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1)


def _get_hand_pose(img: np.ndarray, pose: np.ndarray, prev_hands) -> Tuple[Any, Any]:
    """
    Args:
        img: np.ndarray | represents an image to extract hand pose from
        pose: np.ndarray | represents the body pose
        prev_hands list | represents the previous hand pose. Assumes order [left_hand, right_hand]

    Returns: Tuple[Any, Any]: hand pose with order (left_hand, right_hand)
    """
    left_hand = prev_hands[0]
    right_hand = prev_hands[1]
    wrists = np.concatenate((pose[4, :].reshape(-1, 3), pose[7, :].reshape(-1, 3)), axis=0)
    x_min, x_max = round(np.min(wrists[:, 0])), round(np.max(wrists[:, 0]))
    y_min, y_max = round(np.min(wrists[:, 1])), round(np.max(wrists[:, 1]))
    a1_s = 0 if y_min-OFFSET < 0 else y_min-OFFSET
    a1_f = img.shape[0] if y_max + OFFSET > img.shape[0] else y_max + OFFSET
    a2_s = 0 if x_min-OFFSET < 0 else x_min-OFFSET
    a2_f = img.shape[1] if x_max + OFFSET > img.shape[0] else x_max + OFFSET
    cropped_img = img[a1_s:a1_f, a2_s:a2_f]
    crop_width, crop_height = cropped_img.shape[:2]
    x_offset = y_min - OFFSET
    y_offset = x_min - OFFSET

    with HandLandmarker.create_from_options(options) as landmarker:
        # Perform hand landmarks detection on the provided single image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.array(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)))
        hand_landmarker_result = landmarker.detect(mp_image)
        for i in range(len(hand_landmarker_result.hand_landmarks)):
            conf = hand_landmarker_result.handedness[i][0].score
            tmp_kp = map(lambda k, w=crop_width, h=crop_height, c=conf, x=x_offset, y=y_offset:
                         (k.x*h + y, k.y*w + x, c), hand_landmarker_result.hand_landmarks[i])

            if hand_landmarker_result.handedness[i][0].category_name == 'Left':
                left_hand = np.array(list(tmp_kp))
            elif hand_landmarker_result.handedness[i][0].category_name == 'Right':
                right_hand = np.array(list(tmp_kp))
    left_hand, right_hand = check_hand_pose([left_hand, right_hand], [wrists[1, :],
                                                                      wrists[0, :]], prev_hands)

    # for ij in left_hand:
    #     cv2.circle(img, (round(ij[0]), round(ij[1])), 2, (76, 255, 122), 2)
    # for ij in right_hand:
    #     cv2.circle(img, (round(ij[0]), round(ij[1])), 2, (122, 76, 255), 2)
    # for ij in pose:
    #     cv2.circle(img, (round(ij[0]), round(ij[1])), 1, (255, 255, 255), 1)
    # cv2.imshow("org", img)
    # cv2.imshow("cropped_img", cropped_img)
    # cv2.waitKey(1)

    return left_hand.copy(), right_hand.copy()


def check_hand_pose(hands_pose, wrists_pose, prev_hands):
    for i in range(2):
        if hands_pose[i] is None:
            # If nothing was extracted for this hand, set the pose to be the same as either wrist pose or previous hand
            # pose
            if prev_hands[i] is None:
                hands_pose[i] = np.full((21, 3), wrists_pose[i])
                continue
            else:
                hands_pose[i] = prev_hands[i].copy()

        # Check that the wrist of the current pose is close to the body wrist pose. If distance > 50, we set the hand pose
        # to be the wrist pose. This happens when mp incorrectly predicts left hand to be right hand (or vice-versa)
        if np.linalg.norm(hands_pose[i][0, :2] - wrists_pose[i][:2]) > 20:
            hands_pose[i] = np.full((21, 3), wrists_pose[i])
    return hands_pose


def extract_hand_pose(video_full_path: str, pose_json_path: str, frames_per_clip: int = 30):
    """
    Obtain the hand skeleton and save to output directory
    Args:
        video_full_path: string | path to dir containing video frames
        pose_json_path: str | path to dir containing pose annotations
        frames_per_clip: int | number of frames each clip contains
    """
    logging.info(f"{video_full_path}")
    pose_seq = []
    clips = []
    previous_body_pose = np.full((18, 3), 0)  # Store body skeleton pose from previous frame
    prev_left_h, prev_right_h = None, None
    json_annotations_paths = natsort.natsorted(map(lambda x: os.path.join(pose_json_path, x), os.listdir(pose_json_path)))#[::2]
    video_frames_paths = natsort.natsorted(map(lambda x: os.path.join(video_full_path, x), os.listdir(video_full_path)))#[::2]
    assert all(map(lambda x: os.path.isfile(x), json_annotations_paths)); assert all(map(lambda x: os.path.isfile(x), video_frames_paths))
    assert len(json_annotations_paths) == len(video_frames_paths)
    n_frames = len(video_frames_paths)
    remaining_clips = n_frames % frames_per_clip

    for f, j in zip(video_frames_paths, json_annotations_paths):
        with open(j) as json_file:
            data = json.load(json_file)
            data = data['people']
            if len(data) > 1:
                pose = get_active_person(data, center=(960, 540), min_bbox_area=20000)
            else:
                pose = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence
        pose = pose.reshape(-1, 3)  # format: joints, coordinates
        pose = np.delete(pose, 8, 0)  # remove background pose

        # If keypoint is zero, use previous keypoint location
        if any((pose < 1e-6).flatten()):
            pose[pose < 1e-6] = previous_body_pose[pose < 1e-6]
        previous_body_pose = pose.copy()
        prev_left_h, prev_right_h = _get_hand_pose(cv2.imread(f), pose, [prev_left_h, prev_right_h])
        pose_seq.append(np.concatenate((previous_body_pose, prev_left_h, prev_right_h), axis=0))
        clips.append(f)

        # Check skeleton
        # frame1 = cv2.imread(f)
        # for i, point in enumerate(pose_seq[-1]):
        #     cv2.circle(frame1, (int(point[0]), int(point[1])), 1, (0, 0, 255))
        # cv2.imshow('fra', frame1)
        # cv2.waitKey(0)

    # Ensure the sequence of poses is divisible by self.frames_per_clip
    if remaining_clips != 0:
        while (len(pose_seq) % frames_per_clip) != 0:
            pose_seq.append(pose_seq[-1].copy())  # Repeat the last pose until it is divisible by frames per clip
            clips.append(clips[-1])  # Repeat last clip until it is divisible by frames per clip

    pose_seq = np.array(pose_seq, dtype=np.float32)  # format: frames, joints, coordinates
    clips = np.array(clips)

    # Check same number of pose sequences and check images divisibility of the sequence of poses
    assert (pose_seq.shape[0] == clips.shape[0] and pose_seq.shape[0] >= n_frames and
            (pose_seq.shape[0] % frames_per_clip) == 0)

    return pose_seq, clips


def get_active_person(people, center=(960, 540), min_bbox_area=20000):
    """
       Select the active skeleton in the scene by applying a heuristic of findng the closest one to the center of the frame
       then take it only if its bounding box is large enough - eliminates small bbox like kids
       Assumes 100 * 200 minimum size of bounding box to consider
       Parameters
       ----------
       data : pose data extracted from json file
       center: center of image (x, y)
       min_bbox_area: minimal bounding box area threshold

       Returns
       -------
       pose: skeleton of the active person in the scene (flattened)
       """

    pose = None
    min_dtc = float('inf')  # dtc = distance to center
    for person in people:
        current_pose = np.array(person['pose_keypoints_2d'])
        joints_2d = np.reshape(current_pose, (-1, 3))[:, :2]
        if 'boxes' in person.keys():
            # maskrcnn
            bbox = person['boxes']
        else:
            # openpose
            idx = np.where(joints_2d.any(axis=1))[0]
            bbox = [np.min(joints_2d[idx, 0]),
                    np.min(joints_2d[idx, 1]),
                    np.max(joints_2d[idx, 0]),
                    np.max(joints_2d[idx, 1])]

        A = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # bbox area
        bbox_center = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)  # bbox center

        dtc = np.sqrt(np.sum((np.array(bbox_center) - np.array(center)) ** 2))
        if dtc < min_dtc:
            closest_pose = current_pose
            if A > min_bbox_area:
                pose = closest_pose
                min_dtc = dtc
    # if all bboxes are smaller than threshold, take the closest
    if pose is None:
        pose = closest_pose
    return pose