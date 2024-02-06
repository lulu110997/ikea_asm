"""
TODO: Figure out multiclass labelling
TODO: Figure out why label is sometimes None
TODO: Validate the exported npy data
"""
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import os
import multiprocessing
import statistics
import cv2
import numpy as np
import sqlite3

class DataBase:
    def __init__(self, db_path=None, indexing_files_path=None, set='test'):
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

        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list

        self.video_set = self.get_video_frame_labels()


    def __exit__(self, exc_type, exc_value, traceback):
        self.cursor_vid.close()
        self.cursor_annotations.close()
        self.db.close()

    def save_xy(self):
        """
        Save the pose sequences per frame and corresponding frame labels (as class indices). This function does not use
        joblib (ie performs tasks sequentially/in one process/thread)
        """
        poses2save = []
        labels2save = []
        video_paths = []
        for vid_path, labels, n_frames in tqdm(self.video_set):
            poses = self.load_poses(vid_path, n_frames)  # Obtain the pose sequence
            video_paths.append(video_paths)  # Save video path for validation later on
            # Ensure that we have the same amount of frames for the labels and pose seqs. The last frame may have been
            # duplicated to ensure the sequence of poses for this video is divisible by self.frames_per_clip
            if labels.shape[0] < poses.shape[0]:
                rep_val = poses.shape[0] - labels.shape[1]
                labels = np.hstack((labels, np.tile(labels[:, [-1]], rep_val)))  # Repeat the last labels
            labels = labels.transpose()  # format: n_frames, n_classes
            labels = np.argmax(labels, axis=1)  # Obtain labels as class index. format: n_frames

            poses2save.append(poses)
            labels2save.append(labels)
        for i in range(len(poses2save)):
            poses2save[i] = poses2save[i].reshape(-1, self.frames_per_clip, 18, 3)  # n_samples, n_frames, n_keypoints, coords
            labels2save[i] = labels2save[i].reshape(-1, self.frames_per_clip)  # n_samples, n_frames
            video_paths[i] = (video_paths[i], poses2save[-1].shape[0]*poses2save[-1].shape[1])  # Save str and n_frames
        poses2save = np.concatenate(poses2save, axis=0)
        labels2save = np.concatenate(labels2save, axis=0)
        video_paths = np.array(video_paths)
        np.save(f'X_{self.set}.npy', poses2save)
        np.save(f'y_{self.set}.npy', labels2save)
        np.save(f'vid_paths_{self.set}.npy', video_paths)  # For validating data integrity

    def save_xy_mp(self):
        """
        Save the pose sequences per frame and corresponding frame labels (as class indices) using joblib to improve
        efficiency
        """
        num_cores = multiprocessing.cpu_count()
        # Multi-processing to speed up job
        output = (Parallel(backend='threading', n_jobs=num_cores)
                  (delayed(self._collect_xy)(idx, vid_path, labels, n_frames)
                   for idx, (vid_path, labels, n_frames) in enumerate(self.video_set)))
        poses2save = []
        labels2save = []
        video_paths = []
        for video in output:
            poses2save.append(video[0].reshape(-1, self.frames_per_clip, 18, 3))  # n_samples, n_frames, n_keypoints, coords
            labels = video[1].reshape(-1, self.frames_per_clip)  # n_samples, n_frames
            for l in labels:  # The clip label is the label that occur the most
                label = statistics.multimode(l)
                if len(label) > 1:
                    if 0 in label:  # Remove 'none' label
                        label.remove(0)
                    if 17 in label:  # Remove 'other' label
                        label.remove(17)
                    if len(label) != 1:  # Either empty as it (only) contains [0,7] or frame contains more than 1 label
                        label = [0]
                labels2save.append(label[0])
            video_paths.append((video[2], poses2save[-1].shape[0]*poses2save[-1].shape[1]))  # Save str and n_frames

        poses2save = np.concatenate(poses2save, axis=0)
        labels2save = np.array(labels2save, dtype=np.long)
        video_paths = np.array(video_paths)
        np.save(f'X_{self.set}.npy', poses2save)
        np.save(f'y_{self.set}.npy', labels2save)
        np.save(f'vid_paths_{self.set}.npy', video_paths)  # For validating data integrity

    def _collect_xy(self, idx, vid_path, labels, n_frames):
        """
        Return the pose sequence and corresponding frame labels for the given video_path
        Args:
            idx: int | represents the which video in the list is being processed
            vid_path: string | path to the video/images
            labels: np array | labels which corresponds to the frame labels given as class index
            n_frames: int | number of frames in the video

        Returns: tuple | (poses, labels, vid_path). Each row of the poses represent one frame. The no. of rows may be
        greater than n_frames to ensure divisibility with self.clip_per_frame
        """
        print(f"Processing video #{idx+1}/{len(self.video_set)}")
        poses = self.load_poses(vid_path, n_frames)  # Obtain the pose sequence
        # Ensure that we have the same amount of frames for the labels and pose seqs. The last frame may have been
        # duplicated to ensure the sequence of poses for this video is divisible by self.frames_per_clip
        if labels.shape[0] < poses.shape[0]:
            rep_val = poses.shape[0] - labels.shape[1]
            labels = np.hstack((labels, np.tile(labels[:, [-1]], rep_val)))  # Repeat the last labels

        labels = labels.transpose()  # format: n_frames, n_classes
        labels = np.argmax(labels, axis=1)  # Obtain labels as class index
        return poses, labels, vid_path

    def load_poses(self, video_full_path, n_frames):
        """
        Extracts pose for a specific set of frames. The returned number of sequences in the returned array may have be
        greater than the given n_frames. This is to ensure that each clip is divisible by the self.frames_per_clip
        The path to video must be extracted from the output of get_video_frame_labels as it relies on this structure to
        obtain the path to the openpose predictions.
        Args:
            video_full_path: str | path to video obtained from output of get_video_frame_labels
            n_frames: int | represents how many frames the corresponding video contains

        Returns: np array with shape (frames, joints, coordinates)
        """
        pose_seq = []
        pose_path = video_full_path.replace('/data/ikea_asm_dataset_RGB_top_frames', '/annotations/pose_annotations')
        pose_path = pose_path.replace('/images', '/predictions/pose2d/openpose')
        remaining_clips = n_frames % self.frames_per_clip

        for i in range(n_frames):
            # Obtain the number path to json file (pose annotations for this corresponding video)
            pose_json_filename = os.path.join(pose_path, 'scan_video_' + str(i).zfill(12) + '_keypoints' + '.json')
            with open(pose_json_filename) as json_file:
                data = json.load(json_file)

            # Ensure we obtain the active person's pose sequences
            data = data['people']
            if len(data) > 1:
                pose = self.get_active_person(data, center=(960, 540), min_bbox_area=20000)
            else:
                pose = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence

            pose = pose.reshape(-1, 3)  # format: joints, coordinates
            pose = np.delete(pose, 8, 0)
            pose_seq.append(pose)

            # Ensure the sequence of poses is divisible by self.frames_per_clip
            if i+1 == n_frames and remaining_clips != 0:
                while (len(pose_seq) % self.frames_per_clip) != 0:
                    pose_seq.append(pose)  # Repeat the last pose until it is divisible by frames per clip

            # Check skeleton
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

        # Check divisibility of the sequence of poses
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

    def get_action_id(self, atomic_action_id, object_id):
        """
        find the action id of an atomic action-object pair, returns None if not in the set
        :param atomic_action_id: int id of atomic action
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
            if not os.path.exists(video_full_path):  # check if frame folder exists
                continue

            label = np.zeros((self.num_classes, n_frames), np.float32)  # TODO: dont allow multi-class representation
            label[0, :] = np.ones((1, n_frames), np.float32)  # initialize all frames as background|transition
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


db = DataBase()
db.save_xy_mp()
