import sys

import cv2
import os
import numpy as np

ROOT = '/home/louis/.ikea_asm_2d_pose/openpose'
SPLIT = 1

train_x_path = os.path.join(ROOT, str(SPLIT), 'X_train.npy')
test_x_path = os.path.join(ROOT, str(SPLIT), 'X_test.npy')
vid_path_train = os.path.join(ROOT, str(SPLIT), 'vid_paths_train.npy')

X_train = np.load(train_x_path).reshape((-1, 18, 3))
y_train = np.load(train_x_path.replace('X', 'y')).reshape((X_train.shape[0]))
X_test = np.load(test_x_path)
y_test = np.load(test_x_path.replace('X', 'y'))
vid_train_paths = np.load(vid_path_train)
vid_test_paths = np.load(vid_path_train.replace('train', 'test'))

try:
    for v in vid_train_paths[1:]:
        path = str(v[0])
        n_frames = int(v[1])
        frames_list = os.listdir(path)

        for idx, f in enumerate(frames_list):
            cv2.namedWindow("win", cv2.WINDOW_NORMAL)
            frame_path = os.path.join(path, f)
            frame1 = cv2.imread(frame_path)
            pose = X_train[idx]
            for i, point in enumerate(pose):
                cv2.circle(frame1, (int(point[0]), int(point[1])), 1, (0, 0, 255))
                cv2.putText(frame1, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('win', frame1)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()

finally:
    cv2.destroyAllWindows()