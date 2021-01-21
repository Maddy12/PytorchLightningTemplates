import h5py
import tqdm
import json
import os
import numpy as np
import pickle
import torch
import pandas as pd
import glob

from torch.utils.data import DataLoader
import compress_pickle
import re
from torch._six import container_abcs, string_classes, int_classes


def h5_load_multi(path, dataset_names):
    """
    From https://github.com/noureldien/videograph/blob/master/core/utils.py#L69
    """
    h5_file = h5py.File(path, 'r')
    data = [h5_file[name][()] for name in dataset_names]
    h5_file.close()
    return data


def h5_dump_multi(data, dataset_names, path):
    h5_file = h5py.File(path, 'w')
    n_items = len(data)
    pbar = tqdm.tqdm(range(n_items), total=n_items)
    pbar.set_description("Dumping features to disk")
    for i in pbar:
        item_data = data[i]
        item_name = dataset_names[i]
        h5_file.create_dataset(item_name, data=item_data, dtype=item_data.dtype)
    h5_file.close()


def uniform_sample_frames(frames, n_frames_per_segment, n_frames_per_video=None):
    """
    Extracts 64 segments from frames (either text or video frames).
    From https://github.com/noureldien/videograph
    """
    if n_frames_per_video is None:
        n_frames_per_video = n_frames_per_video
    sampled_frame_pathes = list()
    n_frames = len(frames)
    n_segments = int(n_frames_per_video / n_frames_per_segment)

    if n_frames < n_frames_per_video:
        step = (n_frames - n_frames_per_segment) / float(n_segments)
        idces_start = np.arange(0, n_frames - n_frames_per_segment, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()
    elif n_frames == n_frames_per_video:
        idx = np.arange(n_frames_per_video)
    else:
        step = n_frames / float(n_segments)
        idces_start = np.arange(0, n_frames, step=step, dtype=np.int)
        idx = []
        for idx_start in idces_start:
            idx += np.arange(idx_start, idx_start + n_frames_per_segment, dtype=np.int).tolist()

    v_sampled = frames[idx]
    # sampled_frame_pathes.append(v_sampled)

    # sampled_frame_pathes = np.array(sampled_frame_pathes)
    # return sampled_frame_pathes
    return v_sampled


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def find_closest(A, target):
    """
    https://stackoverflow.com/questions/21388026/find-closest-float-in-array-for-all-floats-in-another-array
    :param A:
    :param target:
    :return:
    """

    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
