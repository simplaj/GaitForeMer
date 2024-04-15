import copy
import json
import math
import os
import pickle
import random

import joblib
import numpy as np
import pandas as pd

from utils import *


# Set path to joint data, label_file, and save path
joints_path = '/root/autodl-tmp/output/' # location of 3d joints 
exam_df = pd.read_csv('label.csv') # label file with example format
save_dir = '/root/autodl-tmp/nhpd/'
print(exam_df.head(10))

# list of valid video names that we can use for gait
video_names = exam_df['Video Name'].tolist()
labels = exam_df['Label'].tolist()
    
# generate a dictionary mapping video name to label
labels_dict = {}
for i in range(len(video_names)):
    labels_dict[video_names[i]] = labels[i]
    
    
def read_keypoints(keypoint_path):
    """
    Read json files in given directory into arrays of pose keypoints. 
    Remove confidence scores and format pose keypoints as a list of tuples, as the preprocessor expects. 
    :param keypoint_path: path to directory of keypoints
    :return: dictionary with <key=video name, value=keypoints>
    """
    pose_dict = {}
    for video_name in video_names:
        vibe_output = joblib.load(keypoint_path + video_name + '/vibe_output.pkl') 
        print('processing video name:', video_name)

        # Choose the subject with pose detected for the most frames. Modify as necessary. 
        max_key = list(vibe_output)[0]
        max_frames = len(vibe_output[max_key]['joints3d'])
        for key in vibe_output:
            num_frames = len(vibe_output[key]['joints3d'])
            if num_frames > max_frames:
                max_frames = num_frames
                max_key = key
        joints3d = vibe_output[max_key]['joints3d']

        # normalize each point by the a single joint
        for i in range(len(joints3d)):
            pelvis = 48
            joints3d[i] = joints3d[i] - joints3d[i][pelvis]
        pose_dict[video_name] = np.stack(joints3d)
    return pose_dict



def normalize_poses(pose_dict):
    """
    Normalize each pose along each axis by video. Divide by the largest value in each direction
    and center around the origin.
    :param pose_dict: dictionary of poses
    :return: dictionary of normalized poses
    """
    normalized_pose_dict = {}
    for video_name in pose_dict:
        poses = pose_dict[video_name].copy()

        maxes = [-1, -1, -1]
        mins = [1, 1, 1]

        for i in range(len(poses)):
            pose = poses[i]
            for j in range(49):
                [x, y, z] = pose[j]
                maxes[0] = max(maxes[0], x)
                maxes[1] = max(maxes[1], y)
                maxes[2] = max(maxes[2], z)
                mins[0] = min(mins[0], x)
                mins[1] = min(mins[1], y)
                mins[2] = min(mins[2], z)
        for i in range(len(poses)):
            pose = poses[i]
            for j in range(49):
                [x, y, z] = pose[j]
                poses[i][j][0] = x / (maxes[0] - mins[0])
                poses[i][j][1] = y / (maxes[1] - mins[1])
                poses[i][j][2] = z / (maxes[2] - mins[2])   
        normalized_pose_dict[video_name] = poses
    return normalized_pose_dict

pose_dict = read_keypoints(joints_path)
normalized_pose_dict = normalize_poses(pose_dict)


def get_clips(video, length, offset):
    """
    Returns a list of partitioned gait segments of given length in frames and offset to next clip
    :param video: input video
    :param length: length of clip
    :param offset: offset between clips
    :return: partition of gait segments into clips
    """
    clips = []
    n = len(video)
    num_clips = math.ceil((n - length+1) / offset)

    i = n
    while i > length:
        temp_end = i
        temp_start = temp_end - length
        clip = np.array(video[temp_start:temp_end])
        clips.append(clip)
        i -= offset
    return clips

def partition_videos(video_names, pose_dict, length=200, offset=50):
    """
    Partition poses from each video into clips.
    :param video_names: names of videos to partition
    :param pose_dict: dictionary of poses for each video
    :param length: length of clip
    :param offset: offset between clips
    :return: dictionary of clips for each video
    """
    clip_dict = {}
    for name in video_names:
        print(name)
        clips = get_clips(pose_dict[name], length, offset)
        clip_dict[name] = clips
    return clip_dict


clip_dict = partition_videos(normalized_pose_dict.keys(), normalized_pose_dict, length=100, offset=30)


def generate_pose_label(clip_dict, test_clip_dict, train_list, val_list, test_list):
    train = {}
    train['pose'] = []
    train['label'] = []
    val = {}
    val['pose'] = []
    val['label'] = []
    test = {}
    test['pose'] = []
    test['label'] = []
    complete_list = train_list + val_list + test_list
    
    # Place each clip in the correct split
    for video_name in train_list:
        clips = test_clip_dict[video_name]
        for clip in clips:
            train['label'].append(labels_dict[video_name])
            train['pose'].append(clip)
    for video_name in val_list + test_list:
        clips = test_clip_dict[video_name]
        for clip in clips:
            if video_name in val_list:
                val['label'].append(labels_dict[video_name])
                val['pose'].append(clip)
            elif video_name in test_list:
                test['label'].append(labels_dict[video_name])
                test['pose'].append(clip)
    print("len train", len(train['label']))
    return train, val, test

def generate_leave_one_out_folds(clip_dict, test_clip_dict, save_dir, seed=None):
    """
    Generate folds for leave-one-out CV.
    :param clip_dict: dictionary of clips for each video
    :param test_clip_dict: dictionary of poses for each test video
    :param save_dir: save directory for folds
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    video_names_list = []
    k = 0
    for video_name in clip_dict:
        video_names_list.append(video_name)
        k += 1
    
    print(video_names_list)
    for j in range(len(video_names)):
        i = j + 1
        train_list = video_names_list[:]
        train_list.remove(video_names[j])
        val_list = []
        test_list = [video_names[j]]
                    
        train, _, test = generate_pose_label(clip_dict, test_clip_dict, train_list, val_list, test_list)
        print("test:", test_list)
        print("train:", len(train['label']))
        print("labels:", test['label'])
        pickle.dump(train_list, open(save_dir+"EPG_train_list_"+str(i)+".pkl", "wb"))
        pickle.dump(test_list, open(save_dir+"EPG_test_list_"+str(i)+".pkl", "wb"))
        pickle.dump(train, open(save_dir+"EPG_train_"+str(i)+".pkl", "wb"))
        pickle.dump(test, open(save_dir+"EPG_test_"+str(i)+".pkl", "wb"))
    pickle.dump(labels_dict, open(save_dir+"EPG_labels.pkl", "wb"))
    
generate_leave_one_out_folds(clip_dict, clip_dict, save_dir, seed=4096)