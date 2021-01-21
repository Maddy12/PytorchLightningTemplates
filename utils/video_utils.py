import os
import cv2
import pickle as pkl
from scipy.io import loadmat
import numpy as np
import imageio
from PIL import Image
import optparse
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.editor import *
import moviepy.editor as moviepyeditor
import paramiko
import tqdm
import glob
import torch
import pdb
import numbers
import random
import librosa
import matplotlib.pyplot as plt
import speech_recognition as sr
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import math
import torch.nn.functional as F
from torch.utils.data import Dataset


def preprocess_img(frame_pth):
    # Loads image in RGB Mode
    img = Image.open(frame_pth)
    img = np.asarray(img).astype(np.float32)

    # normalize such that values range from -1 to 1
    img /= float(127.5)
    img -= 1.0
    return img


class LoadFrames(Dataset):
    def __init__(self, pths):
        """
        This will load the frames of a given video.
        It will load each frame and prepreocess them via normalization.
        :param pths:
        """
        self.pths = pths

    def __len__(self):
        return len(self.pths)

    def __getitem__(self, index):
        frame = self.pths[index]
        return np.array(preprocess_img(frame))


def load_pickle(pth):
    return pickle.load(open(pth, 'rb'))

def get_video_info(video_path):
    """
    Gets FPS, number of frames, and duration of video.
    """
    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = cap.fps
    n_frames = cap.nframes
    duration = cap.duration
    cap.close()
    del cap

    return fps, n_frames, duration


def sample_frames(n_frames, frames, step, window=3, optional=False):
    """
    Create overlapping clips.
    :param int n_frames: Number of frames in video.
    :param list timestamps: Either a list of timestamps for video or indexes for subsetting.
    :param step: Clip length.
    :param int window: Window to slide. If want no overlap between clips, set to 1 because it is `step // window`
    :return:
    """
    start = 0
    end = len(frames)
    idces_start = np.arange(start, end, step=step // window)
    timestamps = np.array(frames)
    idx = []
    for idx_start in idces_start:
        if idx_start + step > n_frames:
            if optional:
                continue
            tmp = np.arange(n_frames - step, n_frames, dtype=np.int).tolist()
            if end in tmp:
                continue
            idx.append(frames[tmp])

        else:
            tmp = np.arange(idx_start, idx_start + step, dtype=np.int).tolist()
            if end in tmp:
                continue
            idx.append(frames[tmp])
    return idx


def sample_diff_fps(n_frames, timestamps, clip_length, fps, sample_fps):
    clips = sample_frames(n_frames, timestamps, fps, window=1, optional=False)
    new_clips = list()
    window = int(np.floor(fps/sample_fps))
    if window == 0:
        print("WARNING: Window was zero with fps {} and sample fps {}, window set to 1".format(fps, sample_fps))
        window = 1
    for clip in clips:
        new_clip = clip[::window]
        if len(new_clip) > sample_fps:
            diff = len(new_clip) - sample_fps
            new_clips.append(np.array(new_clip[:-diff]))
        elif len(new_clip) < sample_fps:
            diff = sample_fps - len(new_clip)
            new_clips.append(np.concatenate([new_clip, clip[-diff:]]))
        else:
            new_clips.append(new_clip)
    new_timestamps = np.concatenate(new_clips)
    if len(new_timestamps) < clip_length:
        if clip_length / len(new_timestamps) > 1:
            new_timestamps = [val for val in new_timestamps for _ in range(clip_length // len(new_timestamps))]
            if len(new_timestamps) > clip_length:
                diff = len(new_timestamps) - clip_length
                new_timestamps = new_timestamps[: -diff]
        if len(new_timestamps) < clip_length:
            diff = clip_length - len(new_timestamps)
            new_timestamps = np.concatenate([new_timestamps, new_timestamps[-diff:]])
    new_clips = sample_frames(n_frames, new_timestamps, clip_length, window=1, optional=False)
    return new_clips


def sample_segments(n_frames, n_frames_per_video, segment_length=8):
    """

    """
    n_segments = int(n_frames_per_video / segment_length)

    assert n_frames_per_video % segment_length == 0
    assert n_frames > segment_length

    if n_frames < n_frames_per_video:
        step = (n_frames - segment_length) / float(n_segments)
        idces_start = np.arange(0, n_frames - segment_length, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()
    elif n_frames == n_frames_per_video:
        idx = np.arange(n_frames_per_video)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + segment_length, dtype=np.int).tolist()

    sampled_frames = np.arange(n_frames)[idx]
    return sampled_frames


def resize_crop_scaled(image, target_height=224, target_width=224):
    """
    https://github.com/noureldien/unsupervised_unit_actions
    """
    # re-scale the image by ratio 3/4 so a landscape or portrait image becomes square
    # then resize_crop it

    # for example, if input image is (height*width) is 400*1000 it will be (400 * 1000 * 3/4) = 400 * 750

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, _ = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))
    else:

        # first, rescale it, only if the rescale won't bring the scaled dimention to lower than target_dim (= 224)
        scale_factor = 3 / 4.0
        if height < width:
            new_width = int(width * scale_factor)
            if new_width >= target_width:
                image = cv2.resize(image, (new_width, height))
        else:
            new_height = int(height * scale_factor)
            if new_height >= target_height:
                image = cv2.resize(image, (width, new_height))

        # now, resize and crop
        height, width, _ = image.shape
        if height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

        # this line is important, because sometimes the cropping there is a 1 pixel more
        height, width, _ = resized_image.shape
        if height > target_height or width > target_width:
            resized_image = cv2.resize(resized_image, (target_height, target_width))

    return resized_image


def load_frames(vid_pth, frame_idces, frame_size=224):
    # Load Frames

    cap = moviepyeditor.VideoFileClip(vid_pth)
    fps = float(cap.fps)
    duration = cap.duration
    n_frames = int(fps * duration)

    loaded_frames = np.zeros((len(frame_idces), frame_size, frame_size, 3))
    for i, idx in enumerate(frame_idces):
        time_sec = idx / fps
        frame = cap.get_frame(time_sec)
        frame = resize_crop_scaled(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        loaded_frames[i] = frame
    cap.reader.close()
    cap.close()
    del cap.reader
    del cap
    return loaded_frames


def convert_mp4_to_wav(vid_pth, save_audio_pth):
    """
    This will convert an mp4 file by loading it and saving the audio as an audio file.
    """
    clip = moviepyeditor.VideoFileClip(vid_pth)
    clip.audio.write_audiofile(save_audio_pth)
    return


def load_frames_from_timestamp(cap, frame_idces, frame_size=224):
    # Load Frames
    loaded_frames = np.zeros((len(frame_idces), frame_size, frame_size, 3))
    for i, time_sec in enumerate(frame_idces):
        frame = cap.get_frame(time_sec)
        frame = resize_crop_scaled(frame, target_height=frame_size, target_width=frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        loaded_frames[i] = frame
    return loaded_frames


def load_frames_from_img(frames, frame_size=224):
    # Load Frames
    loaded_frames = np.zeros((len(frames), frame_size, frame_size, 3))
    for i, frame_pth in enumerate(frames):
        frame = preprocess_img(frame_pth)
        frame = resize_crop_scaled(frame, target_height=frame_size, target_width=frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        loaded_frames[i] = frame
    return loaded_frames


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):

        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i + h, j:j + w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i + th, j:j + tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



def resize_image(img, size=(28, 28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)