import numpy as np
from moviepy.editor import *
import tqdm
import torch
import glob
import time
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import random
import numpy as np
from speech_recognition import RequestError
from PIL import Image
import pdb
import tqdm
import moviepy.editor as moviepyeditor
import pickle
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)  # Add another
top_level = os.path.dirname(parent_dir)
sys.path.insert(0, top_level)
from utils import video_utils
from experiments.i3d.exputils import *


class UCFCrime(Dataset):
    def __init__(self, annot, transform, clip_length=8, video_dir='data/UCFCrime/Videos', train=True, **kwargs):
        super(UCFCrime).__init__()
        self.video_dir = video_dir
        self.activities = get_label_list('ucf_crime')
        self.n_activities = len(self.activities)
        self.n_classes = self.n_activities
        self.transform = transform
        self.annot = annot[0] if train else annot[1]
        # self.vids = [vid for vid in list(self.annot.keys()) if 'Normal' not in vid]
        self.vids = list(self.annot.keys())
        if train:
            random.shuffle(self.vids)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        key = self.vids[index]
        vid = '_'.join(key.split('_')[1:])
        frames = self.annot[key]['clip']
        target = self.annot[key]['label']
        cap = cv2.VideoCapture(os.path.join(self.video_dir, target, vid))

        loaded_frames = list()
        for frame_number in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
            success, image = cap.read()
            if success:
                loaded_frames.append(self.preprocess_img(image))
            else:
                print("ERROR LOADING VIDEO {}".format(key))
                exit()
        cap.release()
        loaded_frames = np.stack(loaded_frames)
        n_frames, H, W, C = loaded_frames.shape
        loaded_frames = self.transform(loaded_frames.reshape((n_frames, H, W, C)))
        n_frames, H, W, C = loaded_frames.shape

        # Get targets
        activities = torch.zeros(self.n_activities)
        activities[np.argwhere(self.activities == target)[0][0]] = 1
        assert torch.sum(activities) > 0, "Something wrong with targets."

        return torch.from_numpy(loaded_frames).contiguous().float().view(C, n_frames, H, W), activities

    @staticmethod
    def preprocess_img(image):
        image = np.asarray(image).astype(np.float32)
        # image = (image / 255.) * 2 - 1
        image = image / 127.5
        image -= 1.0
        return image



########################################################################################################################
#################################   Utility for UCF Crime functions   ##################################################
########################################################################################################################


def get_split_ucfcrime(df):
    """
    Splits training and testing based off of label and video.
    :param df:
    :return:
    """
    labels = df['label'].unique()
    split_ratio = 0.90
    train = list()
    test = list()
    for label in labels:
        vids = df[df['label'] == label]['vid'].values
        n_vids = len(vids)
        n_vids_tr = int(n_vids * split_ratio)
        train += list(vids[:n_vids_tr])
        test += list(vids[n_vids_tr:])
    return train, test


def preprocess_metadata(pth, video_dir, save_dir='datafiles/ucf_crime/', clip_length=64):
    """

    :param pth: 'UCF_Crimes/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt'
    :param video_dir:
    :param save_pth:
    :param clip_length:
    :return:
    """
    names = ['vid', 'label', 'evt1_start_frame', 'evt1_end_frame', 'evt2_start_frame', 'evt2_end_frame']
    df = pd.read_csv(pth, delimiter='\s\s', header=None, names=names)

    # Reduce the number of samples for normal videos
    downloaded_normal = os.listdir(os.path.join(video_dir, 'Normal'))
    mean_diff = np.mean(df['evt1_end_frame'] - df['evt1_start_frame'])
    normal_subset = df[df['vid'].isin(downloaded_normal)][:15]
    df = df[df['label'] != 'Normal']
    df = df.append(normal_subset)

    train, test = get_split_ucfcrime(df)
    annot = dict()
    train_clips = dict()
    test_clips = dict()

    annot['train'] = train
    annot['test'] = test
    for row in tqdm.tqdm(df.iterrows(), total=len(df)):

        vid = row[1]['vid']
        label = row[1]['label']

        vid_pth = os.path.join(video_dir, label, vid)
        cap = cv2.VideoCapture(vid_pth)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        annot[vid] = dict()
        annot[vid]['train'] = True if vid in train else False
        annot[vid]['vid_pth'] = vid_pth
        annot[vid]['n_frames'] = length

        if label == 'Abuse':
            window = clip_length - 1
        elif label in ['Robbery', 'Vandalism', 'RoadAccidents', 'Fighting']:
            window = clip_length // 2
        elif label in ['Burglary']:
            window = 1
        else:
            window = clip_length // 3

        if label != 'Normal':
            frame_start = row[1]['evt1_start_frame']
            frame_end = row[1]['evt1_end_frame']
            # for idx_start in np.arange(frame_start, frame_end, step=3):
            clips = get_clip_sample_ucf_crime(length, frame_start, frame_end, clip_length, window)

            if row[1]['evt2_start_frame'] != -1:
                frame_start = row[1]['evt2_start_frame']
                frame_end = row[1]['evt2_end_frame']
                clips += get_clip_sample_ucf_crime(length, frame_start, frame_end, clip_length, window)
        else:
            if length < mean_diff:
                frame_start = random.randint(1, length // 3)
            else:
                frame_start = random.randint(1, length - int(mean_diff))
            frame_end = frame_start + mean_diff
            clips = get_clip_sample_ucf_crime(length, frame_start, frame_end, clip_length, window)

        annot[vid]['clips'] = clips
        annot[vid]['label'] = label
        for idx, clip in enumerate(clips):
            if vid in train:
                train_clips['clip{}_{}'.format(idx, vid)] = {'clip': clip, 'label': label}
            else:
                test_clips['clip{}_{}'.format(idx, vid)] = {'clip': clip, 'label': label}

    torch.save([train_clips, test_clips], os.path.join(save_dir, 'clips.pickle'))
    torch.save(annot, os.path.join(save_dir, 'annot.pickle'))
    return [train_clips, test_clips]


def get_clip_sample_ucf_crime(n_frames, start, end, step, window=3):
    """
    Create overlapping clips.
    :param n_frames: Number of frames in video.
    :param start: Start frame of event.
    :param end: End frame of event.
    :param step: Clip length.
    :return:
    """
    idces_start = np.arange(start, end, step=step // window, dtype=np.int)
    idx = []
    for idx_start in idces_start:
        if idx_start + step > n_frames:
            idx.append(np.arange(n_frames - step, n_frames, dtype=np.int).tolist())
        else:
            idx.append(np.arange(idx_start, idx_start + step, dtype=np.int).tolist())

    return idx

