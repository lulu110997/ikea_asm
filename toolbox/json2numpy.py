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
import sqlite3

POSE_COCO_BODY_PARTS = [
    {0, "Nose"},
    {1, "Neck"},
    {2, "RShoulder"},
    {3, "RElbow"},
    {4, "RWrist"},
    {5, "LShoulder"},
    {6, "LElbow"},
    {7, "LWrist"},
    {8, "Bkg"},
    {9, "RKnee"},
    {10, "RAnkle"},
    {11, "LHip"},
    {12, "LKnee"},
    {13, "LAnkle"},
    {14, ""},
    {15, "LEye"},
    {16, "REar"},
    {17, "LEar"},
    {18, "REye"},
]

class DataBase:
    def __init__(self, db_path=None, indexing_files_path=None, set='train'):
        dataset_path = '/media/louis/STORAGE/IKEA_ASM_DATASET/data/ikea_asm_dataset_RGB_top_frames'
        default_db_path = '/media/louis/STORAGE/IKEA_ASM_DATASET/annotations/action_annotations/ikea_annotation_db_full'
        default_idx_files_path = '/media/louis/STORAGE/IKEA_ASM_DATASET/indexing_files'
        action_list_filename = 'atomic_action_list.txt'
        action_object_relation_filename = 'action_object_relation_list.txt'
        train_filename = 'train_cross_env.txt'
        test_filename = 'test_cross_env.txt'
        frames_per_clip = 64
        frame_skip = 1
        indexing_files_path = default_idx_files_path if indexing_files_path is None else indexing_files_path

        self.db_path = default_db_path if db_path is None else db_path  # Root dir that includes
        self.dataset_path = dataset_path
        self.set = set
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.cursor_vid = self.db.cursor()
        self.cursor_annotations = self.db.cursor()

        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip

        # indexing files
        self.action_list_filename = os.path.join(indexing_files_path, action_list_filename)
        self.action_object_relation_filename = os.path.join(indexing_files_path, action_object_relation_filename)

        self.train_filename = os.path.join(indexing_files_path, train_filename)
        self.test_filename = os.path.join(indexing_files_path, test_filename)

        # load action list and object relation list and sort them together
        self.action_list = self.get_list_from_file(self.action_list_filename)
        self.ao_list = self.get_action_object_relation_list()

        self.ao_list = [x for _, x in sorted(zip(self.action_list, self.ao_list))]
        self.action_list.sort()
        self.action_list.insert(0, "NA")  #  0 label for unlabled frames

        self.num_classes = len(self.action_list)
        self.trainset_video_list = self.get_list_from_file(self.train_filename)
        self.testset_video_list = self.get_list_from_file(self.test_filename)
        self.all_video_list = self.testset_video_list + self.trainset_video_list

        #### ADD STUFF BELOW IF ITS FROM A CHILD CLASS

        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list

        self.video_set = self.get_video_frame_labels()  # This gives you the labels of your 'set' in one-hot encoding

        # for i in range(0, len(self.video_set), 5):
        #     self.load_poses(self.video_set[i][0], [1,5,10])

        # GT files
        # if action_segments_filename is not None:
        #     self.segment_json_filename = action_segments_filename
        #     self.action_labels = self.get_actions_labels_from_json(self.segment_json_filename, mode='gt')

    def __exit__(self, exc_type, exc_value, traceback):
        self.cursor_vid.close()
        self.cursor_annotations.close()
        self.db.close()

    def save_xy(self):
        poses2save = []
        labels2save = []
        for vid_path, labels, n_frames in tqdm(self.video_set):
            poses = self.load_poses(vid_path, n_frames)
            if labels.shape[0] < poses.shape[0]:
                rep_val = poses.shape[0] - labels.shape[1]
                labels = np.hstack((labels, np.tile(labels[:, [-1]], rep_val)))  # Repeat the last labels
                labels = labels.transpose()
            poses2save.append(poses)
            labels2save.append(labels)
        n_samples = 0
        for i in range(len(poses2save)):
            poses2save[i] = poses2save[i].reshape(-1, self.frames_per_clip, 18, 3)  # n_samples, n_frames, n_keypoints, coords
            labels2save[i] = labels2save[i].reshape(-1, self.frames_per_clip, self.num_classes)  # n_samples, n_frames, n_classes, coords
            n_samples += poses2save[i].shape[0]
        poses2save = np.concatenate(poses2save, axis=0)
        labels2save = np.concatenate(labels2save, axis=0)
        np.save(f'X_{self.set}.npy', poses2save)
        np.save(f'y_{self.set}.npy', labels2save)

    def load_poses(self, video_full_path, n_frames):
        """
        Extracts pose for a specific set of frames. The path to video must be extracted from the output of
        get_video_frame_labels as it relies on this structure to obtain the path to the openpose predictions
        Args:
            video_full_path: str | path to video obtained from output of get_video_frame_labels
            n_frames: int | represents how many frames the corresponding video contains

        Returns: np array with shape (frames, joints, coordinates)

        """
        pose_seq = []
        pose_path = video_full_path.replace('/data/ikea_asm_dataset_RGB_top_frames', '/annotations/pose_annotations')
        pose_path = pose_path.replace('/images', '/predictions/pose2d/openpose')
        a='s'
        remaining_clips = n_frames % self.frames_per_clip
        for i in range(n_frames):
            pose_json_filename = os.path.join(pose_path,
                                              'scan_video_' + str(i).zfill(12) + '_keypoints' + '.json')
            # data = utils.read_pose_json(pose_json_filename)
            with open(pose_json_filename) as json_file:
                data = json.load(json_file)
            data = data['people']
            if len(data) > 1:
                pose = self.get_active_person(data, center=(960, 540), min_bbox_area=20000)
            else:
                pose = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence

            pose = pose.reshape(-1, 3)  # format: joints, coordinates
            pose = np.delete(pose, 8, 0)
            pose_seq.append(pose)
            if i+1 == n_frames and remaining_clips != 0:
                while (len(pose_seq) % self.frames_per_clip) != 0:
                    pose_seq.append(pose)  # Repeat the last pose until it is divisible by frames per clip
            # frame1 = np.zeros((1080, 1920, 3), np.uint8)
            # frame_path = os.path.join(video_full_path.replace('STORAGE/IKEA_ASM_DATASET/data/', 'LaCie/louis/'),
            #                                  str(i).zfill(6) + '.jpg')
            # frame1 = cv2.imread(frame_path)
            # for i, point in enumerate(pose):
            #     cv2.circle(frame1, (int(point[0]), int(point[1])), 1, (0, 0, 255))
            #     cv2.putText(frame1, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
            # cv2.imshow('fra', frame1)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

        # pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        # pose_seq = pose_seq[:, :, 0:2].unsqueeze(-1)  # format: frames, joints, coordinates, N_people
        # pose_seq = pose_seq.permute(2, 0, 1, 3)  # format: coordinates, frames, joints, N_people
        pose_seq = np.array(pose_seq, dtype=np.float32)  # format: frames, joints, coordinates

        assert pose_seq.shape[0] >= n_frames and (pose_seq.shape[0] % self.frames_per_clip) == 0

        return pose_seq

    def get_active_person(self, people, center=(960, 540), min_bbox_area=20000):
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

    def compute_skeleton_distance_to_center(self, skeleton, center=(960, 540)):
        """
        Compute the average distance between a given skeleton and the cetner of the image
        Parameters
        ----------
        skeleton : 2d skeleton joint poistiions
        center : image center point

        Returns
        -------
            distance: the average distance of all non-zero joints to the center
        """
        idx = np.where(skeleton.any(axis=1))[0]
        diff = skeleton - np.tile(center, [len(skeleton[idx]), 1])
        distances = np.sqrt(np.sum(diff ** 2, 1))
        mean_distance = np.mean(distances)

        return mean_distance

    def get_list_from_file(self, filename):
        """
        retrieve a list of lines from a .txt file
        :param :
        :return: list of atomic actions
        """
        with open(filename) as f:
            line_list = f.read().splitlines()
        # line_list.sort()
        return line_list

    def get_action_object_relation_list(self):
        # read the action-object relation list file and return a list of integers
        with open(self.action_object_relation_filename, 'r') as file:
            a_o_relation_list = file.readlines()
            a_o_relation_list = [x.rstrip('\n').split() for x in a_o_relation_list]
            a_o_relation_list = [[int(y) for y in x] for x in a_o_relation_list]
        return a_o_relation_list

    def get_video_name_from_id(self, video_id):
        """
        return video name for a given video id
        :param video_id: id of video

        :return: video_name: name of the video
        """
        rows = self.cursor_vid.execute('''SELECT * FROM videos WHERE id = ?''',
                                       (video_id, )).fetchall()
        if len(rows) > 1:
            raise ValueError("more than one video with the desired specs - check database")
        else:
            # output = os.path.join(rows[0]["furniture"], rows[0]["video_name"])
            output = rows[0]["video_path"].split('dev', 1)[0][:-1]
        return output

    def get_annotated_videos_table(self):
        """
        fetch the annotated videos table from the database using dev3
        :return: annotated videos table from dev3
        """
        return_table = self.cursor_vid.execute('''SELECT * FROM videos WHERE annotated = 1 AND camera = ?''',
                                               ("dev3",))
        return return_table

    def get_video_annotations_table(self, video_idx):
        """
        fetch the annotation table of a specific video
        :param :  video_idx: index of the desired video
        :return: video annotations table
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations WHERE video_id = ?''', (video_idx,))

    def get_video_table(self, video_idx):
        """
        fetch the video information row from the video table in the database
        :param :  video_idx: index of the desired video
        :return: video information table row from the databse
        """
        return self.cursor_annotations.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx,))

    def get_annotation_table(self):
        """
        :return: full annotations table (for all videos)
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations ''')

    def get_action_id(self, atomic_action_id, object_id):
        """
        find the action id of an atomic action-object pair, returns None if not in the set
        :param action_id: int id of atomic action
        :param object_id: int id of object
        :return: int compound action id | None if not in set
        """
        idx = None
        for i, ao_pair in enumerate(self.ao_list):
            if ao_pair[0] == atomic_action_id and ao_pair[1] == object_id:
                idx = i + 1  # +1 to allow the NA first action label
                break
        return idx

    def get_video_frame_labels(self):
        """
        Extract the label data of each frame in the video from the database
        TODO: DO NOT ALLOW MULTILABELLING
        TODO: LABELLING IS (NUM_CLASSES, FRAMES) and is a one hot vector type of labelling
        Returns: dataset structure of (video_path, multi-label per-frame, number of frames in the video)
        """
        video_table = self.get_annotated_videos_table()  # Gets table for ALL dev3 videos
        vid_list = []
        for row in video_table:
            n_frames = int(row["nframes"])
            video_path = row['video_path']
            video_name = os.path.join(video_path.split('/')[0], video_path.split('/')[1])
            video_full_path = os.path.join(self.dataset_path, video_path)

            if not video_name in self.video_list:
                continue
            if n_frames < 66 * self.frame_skip:  # check video length
                continue
            # if not os.path.exists(video_full_path):  # check if frame folder exists
            #     continue

            label = np.zeros((self.num_classes, n_frames), np.float32) # TODO: dont allow multi-class representation
            label[0, :] = np.ones((1, n_frames), np.float32)   # initialize all frames as background|transition
            video_id = row['id']
            annotation_table = self.get_video_annotations_table(video_id)
            for ann_row in annotation_table:
                atomic_action_id = ann_row["atomic_action_id"]  # map the labels
                object_id = ann_row["object_id"]
                action_id = self.get_action_id(atomic_action_id, object_id)
                # if not isinstance(action_id, int):
                #     print(atomic_action_id, object_id) # TODO: check why action id is None with these inputs
                end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] < n_frames else n_frames
                if action_id is not None:
                    label[:, ann_row['starting_frame']:end_frame] = 0  # remove the background label
                    label[action_id, ann_row['starting_frame']:end_frame] = 1
            vid_list.append((video_full_path, label, n_frames))
        return vid_list

    def get_actions_labels_from_json(self, json_filename, device='dev3', mode='gt'):
        """

         Loads a label segment .json file (ActivityNet format
          http://activity-net.org/challenges/2020/tasks/anet_localization.html) and converts to frame labels for evaluation

        Parameters
        ----------
        json_filename : output .json file name (full path)
        device: camera view to use
        Returns
        -------
        frame_labels: one_hot frame labels (allows multi-label)
        """
        labels = []
        with open(json_filename, 'r') as json_file:
            json_dict = json.load(json_file)

        if mode == 'gt':
            video_results = json_dict["database"]
        else:
            video_results = json_dict["results"]
        for scan_path in video_results:
            video_path = os.path.join(scan_path, device, 'images')
            video_idx = self.get_video_id_from_video_path(video_path, device=device)
            video_info = self.get_video_table(video_idx).fetchone()
            n_frames = video_info["nframes"]
            current_labels = np.zeros([n_frames, self.num_classes])
            if mode == 'gt':
                segments = video_results[scan_path]['annotation']
            else:
                segments = video_results[scan_path]
            for segment in segments:
                action_idx = self.action_list.index(segment["label"])
                start = segment['segment'][0]
                end = segment['segment'][1]
                current_labels[start:end, action_idx] = 1
            labels.append(current_labels)
        return labels

    def get_video_id_from_video_path(self, video_path, device='dev3'):
        """
        return video id for a given video path
        :param video_path: path to video (including device and image dir)
        :return: video_id: id of the video
        """
        rows = self.cursor_vid.execute('''SELECT * FROM videos WHERE video_path = ? AND camera = ?''',
                                       (video_path, device)).fetchall()
        if len(rows) > 1:
            raise ValueError("more than one video with the desired specs - check database")
        else:
            output = rows[0]["id"]
        return output

db = DataBase()
db.save_xy()
sys.exit()

sys.exit()
    # db.get_video_annotations_table(vid_id)
# labels = db.get_actions_labels_from_json(results_json, mode='gt')

# for row in video_table:
#     video_id = row['id']
#     video_name = db.get_video_name_from_id(video_id)
#     # This just gives the different actions that were performed in the video but
#     # does not include timestamps of action
#     video_annotation_table = db.get_video_annotations_table(video_id).fetchall()
#     print(len(video_annotation_table))
#     for ann_row in video_annotation_table:
#         action_id = db.get_action_id(ann_row['atomic_action_id'], ann_row['object_id'])
#         print(action_id)
