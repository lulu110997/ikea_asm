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
import typing
from typing import Tuple, Union, Any

from numpy import ndarray, _DType_co, dtype, _ShapeType, void
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
MILLI = 1000
OFFSET = (10, 10)

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


def crop_img(img: np.ndarray, pose: np.ndarray) -> Tuple[...]:
    """
    Args:
        img (np.ndarray): represents
        pose (np.ndarray):

    Returns: Tuple[...] : me
    """

    x_min = 100000
    x_max = 0
    y_min = 100000
    y_max = 0

    for i in pose:
        if i[0, 0] < x_min:
            x_min = i[0, 0]
        if i[0, 1] < y_min:
            y_min = i[0, 1]

        if i[0, 0] > x_max:
            x_max = i[0, 0]
        if i[0, 1] > y_max:
            y_max = i[0, 1]

    img_offsets = (y_min-OFFSET[0], y_max+OFFSET[0], x_min-OFFSET[1], x_max+OFFSET[1])
    img = img[y_min-OFFSET[0]:y_max+OFFSET[0], x_min-OFFSET[1]:x_max+OFFSET[1]]

    return (img_offsets, img)  # noqa


def save_hand_skeleton(filename):
    """
    Obtain the hand skeleton and save to output directory
    Args:
        filename: string | path to image
    """
    with HandLandmarker.create_from_options(options) as landmarker:
        pose = ...
        cv_image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        cv_image = crop_img(cv_image, pose)
        cv2.imshow('win', cv_image)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        # print(filename); sys.exit()

        # Perform hand landmarks detection on the provided single image.
        # The hand landmarker must be created with the image mode.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cv_image))
        hand_landmarker_result = landmarker.detect(mp_image)
        print(hand_landmarker_result); sys.exit()

    output_filename = filename.replace(INPUT_PATH, OUTPUT_PATH)
    output_dir_path = os.path.dirname(os.path.abspath(output_filename))
    if not os.path.exists(output_filename):
        os.makedirs(output_dir_path, exist_ok=True)

    # Save result
    json_file = convert2json(hand_landmarker_result)


def convert2json(result):
    """
    Convert hand landmarker result from mediapipe to have similar structure as openpose json file
    Args:
        result: mediapipe's hand landmarker result | output of fn for detecting the hand skeleton
    """
    # z (depth) can be extracted from the depth images
    # Need the form xn, yn, cn. Use only the un-normalised projected values
    print(result.handedness)
    # sys.exit()
    json_data = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": []
            }
        ]
    }

    return json_data

file_list = utils.get_list_of_all_files(INPUT_PATH, file_type='.jpg')
print(len(file_list))
# for i in file_list[600:]:
#     save_hand_skeleton(i)

mpose_sample = np.load('/home/louis/.mpose/openpose/1/X_test.npy')
# print(mpose_sample[0].shape, mpose_sample[0,0,:,:])
seq_list =[]
for seq in mpose_sample:
    v1 = np.zeros((30 + 1, seq.shape[1], 3 - 1))
    v2 = np.zeros((30 + 1, seq.shape[1], 3 - 1))
    v1[1:, ...] = seq[:, :, :2]
    v2[:30, ...] = seq[:, :, :2]
    vel = (v2 - v1)[:-1, ...]
    data = np.concatenate((seq[:, :, :2], vel), axis=-1)
    data = np.concatenate((data, seq[:, :, -1:]), axis=-1)
    seq_list.append(data)
X_train = np.stack(seq_list)
s='asd'
with open('/media/louis/STORAGE/IKEA_ASM_DATASET/annotations/pose_annotations/Kallax_Shelf_Drawer/'
          '0001_black_table_02_01_2019_08_16_14_00/dev3/predictions/pose2d/openpose/'
          'scan_video_000000000000_keypoints.json') as file:
    ikea_openpose = json.load(file)['people'][0]['pose_keypoints_2d']
# print(len(ikea_openpose)/3)
# tmp = [995.1219512195122, 228.91304347826087, 0.9787825345993042,
#        1030.2439024390244, 270.0, 0.9344065189361572,
#        965.8536585365854, 275.8695652173913, 0.8997037410736084,
#        945.3658536585366, 369.7826086956522, 0.9310221076011658,
#        954.1463414634146, 451.95652173913044, 0.9304882884025574,
#        1091.7073170731708, 267.0652173913043, 0.936791181564331,
#        1123.90243902439, 360.9782608695652, 0.9447482228279114,
#        1115.1219512195123, 451.95652173913044, 0.9820445775985718,
#        0, 0, 0,
#        1000.9756097560976, 451.95652173913044, 0.748716413974762,
#        1021.4634146341463, 584.0217391304348, 0.806501030921936,
#        1027.3170731707316, 669.1304347826086, 0.5967968702316284,
#        1085.8536585365853, 446.0869565217391, 0.7661957144737244,
#        1088.780487804878, 578.1521739130435, 0.7979868650436401,
#        1094.6341463414633, 666.1956521739131, 0.593521773815155,
#        992.1951219512194, 214.2391304347826, 1.064151406288147,
#        1012.6829268292684, 217.17391304347828, 0.9416921138763428,
#        998.0487804878048, 202.5, 0.7074303030967712,
#        1047.8048780487807, 211.30434782608697, 0.9728320240974426]


# # compress each scan individually and maintain directory tree structure - parallel to speed it up
# with Pool(8) as p:
#   list(tqdm(p.imap(get_hand_skeleton, file_list), total=len(file_list)))


# def shrink_and_save(filename):
#     # saves and shrinks a file from the dataset.
#     # if the file already exists in the output directory - skip it.
#
#     output_filename = filename.replace(INPUT_PATH, OUTPUT_PATH)
#     output_dir_path = os.path.dirname(os.path.abspath(output_filename))
#     if not os.path.exists(output_filename):
#         os.makedirs(output_dir_path, exist_ok=True)
#
#         if FLAGS.mode == 'depth':
#             img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
#             img = img * 255 / 4500
#             # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#             img = cv2.resize(img, dsize=(int(img.shape[0] / SHRINK_FACTOR), int(img.shape[1] / SHRINK_FACTOR)))
#             cv2.imwrite(output_filename, img)
#         else:
#             # shrink and save the data
#             img = Image.open(filename)
#             img = img.resize((int(img.size[0] / SHRINK_FACTOR), int(img.size[1] / SHRINK_FACTOR)))
#             img.save(output_filename)


