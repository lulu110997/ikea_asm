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
import json
from typing import Tuple, Union, Any
from natsort import natsort
from tqdm import tqdm
import argparse
import tb_utils as utils
import os
from PIL import Image
from multiprocessing import Pool
import cv2
import numpy as np
import sys
import mediapipe as mp
# import logging
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
# logging.getLogger('mediapipe').setLevel(logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Flags for postprocessing exports
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str,
                    default='/media/louis/STORAGE/IKEA_ASM_DATASET/data/ikea_asm_dataset_RGB_top_frames',
                    help='path to the ANU IKEA dataset')
parser.add_argument('--output_path', type=str,
                    default='/media/louis/STORAGE/IKEA_ASM_DATASET/data/tmp',
                    help='path to the ANU IKEA dataset')
FLAGS = parser.parse_args()

INPUT_PATH = FLAGS.input_path
OUTPUT_PATH = FLAGS.output_path

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

PATH_TO_MODEL = '/home/louis/Data/Fernandez_HAR/hand_landmarker.task'
OFFSET = (50, 50)  # (y,x)

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
def _get_hand_pose(img: np.ndarray, pose: np.ndarray) -> Tuple[Any]:
    """
    Args:
        img (np.ndarray): represents
        pose (np.ndarray):

    Returns: Tuple[...] : me
    """
    assert pose.shape == (18, 3)
    left_hand = None
    right_hand = None
    wrists = np.concatenate((pose[3, :].reshape(-1,3), pose[3, :].reshape(-1,3)), axis=0)
    x_min, x_max = round(np.min(wrists[:, 0])), round(np.max(wrists[:, 0]))
    x_offset = 0
    y_min, y_max = round(np.min(wrists[:, 1])), round(np.max(wrists[:, 1]))
    y_offset = img.shape[1] - (y_max+OFFSET[1])
    l_cropped_img = img[y_min-OFFSET[0]:y_max+OFFSET[0], x_min-OFFSET[1]:x_max+OFFSET[1]]
    r_cropped_img = img[y_min-OFFSET[0]:y_max+OFFSET[0], x_min-OFFSET[1]:x_max+OFFSET[1]]
    crop_width, crop_height = cropped_img.shape[:2]
    cv2.imshow("cropped_img", cropped_img)
    for ij in pose:
        cv2.circle(img, (round(ij[0]), round(ij[1])), 3, (255, 255, 255), 1)
    # cv2.imshow("org", img) TODO: check img size of mediapipe, use images that have hand skeleton extracted
    cv2.waitKey(1)

    with HandLandmarker.create_from_options(options) as landmarker:
        # Perform hand landmarks detection on the provided single image.
        # The hand landmarker must be created with the image mode.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cropped_img))
        hand_landmarker_result = landmarker.detect(mp_image)
        print(len(hand_landmarker_result.hand_landmarks))
        for i in range(len(hand_landmarker_result.hand_landmarks)):
            conf = hand_landmarker_result.handedness[i][0].score
            tmp_kp = map(lambda k, w=crop_width, h=crop_height, c=conf, x=x_offset, y=y_offset:
                         (k.x*h + y, k.y*w, c), hand_landmarker_result.hand_landmarks[i])

            if hand_landmarker_result.handedness[i][0].category_name == 'Left':
                left_hand = np.array(list(tmp_kp))
            elif hand_landmarker_result.handedness[i][0].category_name == 'Right':
                right_hand = np.array(list(tmp_kp))
        # if len(hand_landmarker_result.hand_landmarks) > 1:
        #     print(len(hand_landmarker_result.hand_landmarks))
        #     for ij in left_hand:
        #         cv2.circle(img, (round(ij[0]), round(ij[1])), 3, (255, 255, 255), 1)
        #     for ij in right_hand:
        #         cv2.circle(img, (round(ij[0]), round(ij[1])), 3, (122, 76, 255), 1)
        #     cv2.imshow("org", img)
        #     cv2.waitKey(0)
            p=1
    # cv2.imshow("cropped_img", cropped_img)
    # for ij in pose:
        # cv2.circle(img, (round(ij[0]), round(ij[1])), 3, (255, 255, 255), 1)

    # output_filename = filename.replace(INPUT_PATH, OUTPUT_PATH)
    # output_dir_path = os.path.dirname(os.path.abspath(output_filename))
    # if not os.path.exists(output_filename):
    #     os.makedirs(output_dir_path, exist_ok=True)
    #
    # # Save result
    # json_file = convert2json(hand_landmarker_result)
    #
    # return (img_offsets, img)  # noqa


def extract_hand_pose(video_full_path: str, pose_json_path: str):
    """
    Obtain the hand skeleton and save to output directory
    Args:
        video_full_path: string | path to dir containing video frames
        pose_json_path: str | path to dir containing pose annotations
    """
    previous_body_pose = None  # Store body skeleton pose from previous frame
    json_annotations_paths = natsort.natsorted(map(lambda x: os.path.join(pose_json_path, x), os.listdir(pose_json_path)))#[45:]
    video_frames_paths = natsort.natsorted(map(lambda x: os.path.join(video_full_path, x), os.listdir(video_full_path)))#[45:]
    assert all(map(lambda x: os.path.isfile(x), json_annotations_paths)); assert all(map(lambda x: os.path.isfile(x), video_frames_paths))
    assert len(json_annotations_paths) == len(video_frames_paths)
    n_frames = len(video_frames_paths)
    remaining_clips = n_frames % 30 #self.frames_per_clip  # TODO: change this to attr
    # cv2.namedWindow("org"); cv2.namedWindow("cropped_img"); input()

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
            pose[pose < 1e-6] = previous_poses[pose < 1e-6]
        previous_poses = pose.copy()  # TODO: make an attr
        img = cv2.imread(f)
        _get_hand_pose(img, pose)

    cv2.destroyAllWindows()
    print("done")


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