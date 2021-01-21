import os
import torch
from torch.utils.data import Dataset
import compress_pickle
import baker
import glob
import tqdm
import numpy as np
import random
import pdb
import zlib
from PIL import Image
from torchvision.transforms import Compose
import re
from multiprocessing import cpu_count
from torchvideotransforms import video_transforms, volume_transforms
from torchvision import transforms

# Local
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import video_utils
from utils import audio_utils


class Breakfast(Dataset):
    def __init__(self, clips, transform, data_dir, eval=False, debug=False, train=True, binary_tgt=True, **kwargs):
        """
        Breakfast videos have a fps of 15.
        :param clips: Loaded split
        :param transform:
        :param data_dir:
        :param eval:
        :param debug:
        :param train:
        :param binary_tgt:
        :param kwargs:
        """
        self.args = kwargs
        self.clips = clips
        self.transform = transform
        self.data_dir = data_dir
        self.n_activities = 48 + 2
        self.n_classes = 10
        self.eval = eval
        self.debug = debug
        self.train = train
        self.binary_tgt = binary_tgt
        self.clip_length = self.args['clip_length']
        assert os.path.exists('datafiles/breakfast/activities_list.pt'), \
            "Make sure you have the list of classes saved in `datafiles/breakfast/activities_list.pt`"
        self.labels_list = np.array(torch.load('datafiles/breakfast/activities_list.pt'))
        if not self.debug:
            clips = os.path.join(clips, '{}_clips_{}frames.pt'.format('training' if train else 'testing',
                                                                      self.clip_length))
            self.clips = torch.load(clips)
            # if train:
            #     k = 2500
            #     print("Sub-sampling randomly for k={}".format(k))
            #     self.clips = random.sample(self.clips[1:], k=k)
        else:
            self.clips = np.arange(64)

        # For a multiple predictions
        if 'dualhead' in self.args.keys():
            self.dualhead = self.args['dualhead']
            if self.dualhead:
                assert os.path.exists('datafiles/breakfast/unit_actions_list.pt'), \
                    "Make sure you have the list of unit-classes saved in `datafiles/breakfast/unit_actions_list.pt`"
                self.sub_activities = np.array(
                    torch.load('datafiles/breakfast/unit_actions_list.pt') + ['walk_in', 'walk_out'])
        else:
            self.dualhead = False

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        """

        :param index: ex. 2382, 27368, 756, 31454, 33829
        :return:
        """
        if self.debug:
            n_frames = self.clip_length
            loaded_frames = torch.rand(3, n_frames, 224, 224)
            if self.binary_tgt:
                activities = torch.zeros(self.n_classes)
                activities[random.randint(1, self.n_classes - 1)] = 1
            else:
                activities = random.randint(1, self.n_classes)

            if self.dualhead:
                activities = [np.random.randint(0, 2, size=(self.n_activities, self.clip_length)).astype(float),
                              activities]
        else:

            try:
                clip = self.clips[index]
                loaded_frames = self.load_frames(clip)

            except Exception as e:
                print("Failed to load frames {}: {}".format(self.clips[index]['pth'], e))
                try_count = 0
                while try_count < 10:
                    loaded_frames = self.select_random_clip()
                    if loaded_frames is not None:
                        break
                    try_count += 1
                else:
                    raise Exception("Loading frames failed after 10 attempts.")

            target = clip['ce_id']
            if self.binary_tgt:
                classes = torch.zeros(self.n_classes)
                classes[np.argwhere(self.labels_list == target)[0][0]] = 1
                assert torch.sum(classes) > 0, "Something wrong with targets."
            else:
                classes = np.argwhere(self.labels_list == target)[0][0]

            if self.dualhead:
                subtarget = clip['ce_steps']
                activities = self.get_dualhead(subtarget, loaded_frames.shape[1], classes)
            else:
                activities = classes

        return loaded_frames, activities

    def select_random_clip(self):
        print("Selecting random clip...")
        try:
            clip = random.choice(self.clips)
            loaded_frames = self.load_frames(clip)
            return loaded_frames
        except (zlib.error, ValueError) as e:
            print('%s %s: %s' % (clip['pth'], type(e), e))
            return None

    def load_frames(self, clip):
        loaded_frames = compress_pickle.load(clip['pth'])
        if loaded_frames.shape[1] == 3:
            T, C, H, W = loaded_frames.shape
            loaded_frames = loaded_frames.reshape(T, H, W, C)
        loaded_frames = self.transform(
            [Image.fromarray(np.uint8(frame), mode='RGB') for frame in loaded_frames])
        loaded_frames = loaded_frames.contiguous().float()  # .view(C, n_frames, H, W)
        return loaded_frames

    def get_dualhead(self, subtarget, n_frames, classes):
        if self.binary_tgt:
            subclasses = torch.zeros(self.n_activities, n_frames)
            # Iterate through each frame and set the target
            for idx, sub in enumerate(subtarget):
                try:
                    subclasses[int(sub)][idx] = 1
                except:
                    print(subtarget)
        else:
            subclasses = subtarget
        activities = [subclasses, classes]
        return activities


class BreakfastNodewise(Dataset):
    def __init__(self, clips_dir, annot_pth, clip_length=512, train=True, debug=False,
                 **kwargs):
        """

        :param clips_dir: Ex. ../../c3-0/datasets/Breakfast/frames/
        :param annot_pth: Ex. datafiles/breakfast/annot_activities.pt
        :param audio_norm_pth:
        :param clip_length:
        :param train:
        :param debug:
        :param kwargs:
        """
        self.args = kwargs
        self.debug = debug
        self.training = train
        self.train = train

        self.frames_dir = clips_dir
        assert os.path.exists('datafiles/breakfast/split.pt'), "Ensure you have generated a train/test split."
        self.split = torch.load('datafiles/breakfast/split.pt')
        self.split = self.split[0] if train else self.split[1]
        if not debug:
            self.video_list = glob.glob(os.path.join(clips_dir, '*', ))
            self.video_list = [pth for pth in self.video_list if pth.split('/')[-1] in self.split]
        else:
            self.video_list = self.split

        self.clip_length = clip_length
        self.dim = 128
        self.crop_dim = 116
        self.sample_fps = 10
        self.fps = 15

        self.n_activities = 48 + 2  # this is for unit actions
        self.n_classes = 10

        assert os.path.exists('datafiles/breakfast/activities_list.pt'), \
            "Make sure you have the list of classes saved in `datafiles/breakfast/activities_list.pt`"
        self.labels_list = np.array(torch.load('datafiles/breakfast/activities_list.pt'))
        self.annot = torch.load(annot_pth)

        if not train:
            self.video_transform = transforms.Compose([video_transforms.CenterCrop(self.crop_dim),
                                                       volume_transforms.ClipToTensor()])
        else:
            self.video_transform = transforms.Compose([video_transforms.RandomResizedCrop(self.crop_dim),
                                                       video_transforms.RandomHorizontalFlip(),
                                                       # video_transforms.ColorJitter(brightness=.5, contrast=.5,
                                                       #                              saturation=.5, hue=.25),
                                                       volume_transforms.ClipToTensor()
                                                       ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        if self.debug:
            n_frames = 64
            loaded_frames = torch.rand(3, n_frames, self.crop_dim, self.crop_dim)
            target = 10
            return loaded_frames, target

        pth = self.video_list[index]
        uid = pth.split('/')[-1]
        n_frames = len(os.listdir(pth))
        frames = np.array(glob.glob(os.path.join(pth, '*')))
        frames.sort()
        frames = video_utils.sample_diff_fps(n_frames, frames, self.clip_length, self.fps, self.sample_fps)[0]
        loaded_frames = video_utils.load_frames_from_img(frames, frame_size=self.dim)
        # C, T, H, W = loaded_frames.shape
        # loaded_frames = loaded_frames.reshape(T, H, W, C)
        loaded_frames = self.video_transform(loaded_frames)
        try:
            target = self.annot[uid]
        except KeyError:
            target = [i for i, label in enumerate(self.labels_list) if label in uid][0]

        return loaded_frames, target


class BreakfastTriplet(Dataset):
    def __init__(self, clips_dir, annot_pth, clip_length=512, train=True, debug=False,
                 **kwargs):
        """

        :param clips_dir: Ex. ../../c3-0/datasets/Breakfast/frames/
        :param annot_pth: Ex. datafiles/breakfast/annot_activities.pt
        :param audio_norm_pth:
        :param clip_length:
        :param train:
        :param debug:
        :param kwargs:
        """
        self.args = kwargs
        self.debug = debug
        self.training = train
        self.train = train

        self.frames_dir = clips_dir
        assert os.path.exists('datafiles/breakfast/split.pt'), "Ensure you have generated a train/test split."
        self.split = torch.load('datafiles/breakfast/split.pt')
        self.split = self.split[0] if train else self.split[1]
        if not debug:
            self.video_list = glob.glob(os.path.join(clips_dir, '*', ))
            self.video_list = [pth for pth in self.video_list if pth.split('/')[-1] in self.split]
        else:
            self.video_list = self.split

        self.clip_length = clip_length
        self.dim = 128
        self.crop_dim = 116
        self.sample_fps = 10
        self.fps = 15

        self.n_activities = 48 + 2  # this is for unit actions
        self.n_classes = 10

        assert os.path.exists('datafiles/breakfast/activities_list.pt'), \
            "Make sure you have the list of classes saved in `datafiles/breakfast/activities_list.pt`"
        self.labels_list = np.array(torch.load('datafiles/breakfast/activities_list.pt'))
        self.annot = torch.load(annot_pth)
        self.target_dict = self.generate_target_dict()

        if not train:
            self.video_transform = transforms.Compose([video_transforms.CenterCrop(self.crop_dim),
                                                       volume_transforms.ClipToTensor()])
        else:
            self.video_transform = transforms.Compose([video_transforms.RandomResizedCrop(self.crop_dim),
                                                       video_transforms.RandomHorizontalFlip(),
                                                       # video_transforms.ColorJitter(brightness=.5, contrast=.5,
                                                       #                              saturation=.5, hue=.25),
                                                       volume_transforms.ClipToTensor()
                                                       ])

    def generate_target_dict(self):
        target_dict = dict()
        for label in self.labels_list:
            target_dict[label] = [pth for pth in self.video_list if label in pth]
        return target_dict

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        anchor_loaded_frames, anchor_target = self.get_frames(self.video_list[index])
        positive_loaded_frames, _ = self.get_frames(random.choice(self.target_dict[anchor_target[1]]))
        negative_label = random.choice([label for label in self.target_dict.keys() if label != anchor_target[1]])
        negative_loaded_frames, negative_target = self.get_frames(random.choice(self.target_dict[negative_label]))
        loaded_frames = [anchor_loaded_frames, positive_loaded_frames, negative_loaded_frames]
        loaded_targets = [anchor_target[0], anchor_target[0], negative_target[0]]
        return loaded_frames, loaded_targets

    def get_frames(self, pth):
        uid = pth.split('/')[-1]
        n_frames = len(os.listdir(pth))
        frames = np.array(glob.glob(os.path.join(pth, '*')))
        frames.sort()
        frames = video_utils.sample_diff_fps(n_frames, frames, self.clip_length, self.fps, self.sample_fps)[0]
        loaded_frames = video_utils.load_frames_from_img(frames, frame_size=self.dim)
        # C, T, H, W = loaded_frames.shape
        # loaded_frames = loaded_frames.reshape(T, H, W, C)
        loaded_frames = self.video_transform(loaded_frames)
        target = [(i, label) for i, label in enumerate(self.labels_list) if label in uid][0]

        return loaded_frames, target


def test_dataset(train):
    metadata_pth = '/home/schiappa/PytorchLightningTemplates/datafiles/breakfast'
    checkpoint_pth = '/home/schiappa/PytorchLightningTemplates/experiments/i3d/checkpoints'
    videos_dir = '/home/c3-0/datasets/Breakfast/videos'
    backbone_model = '/home/schiappa/PytorchLightningTemplates/backbone_models/i3d_model_rgb.pth'
    data_dir = '/home/schiappa/PytorchLightningTemplates/datafiles/breakfast'
    #
    if not train:
        transform = transforms.Compose([video_transforms.CenterCrop(224), volume_transforms.ClipToTensor()])
    else:
        transform = transforms.Compose([video_transforms.RandomResizedCrop(224),
                                        video_transforms.RandomHorizontalFlip(),
                                        video_transforms.ColorJitter(brightness=.5, contrast=.5,
                                                                     saturation=.5, hue=.25),
                                        volume_transforms.ClipToTensor()
                                        ])
    dataset = Breakfast(metadata_pth, clip_length=64, transform=transform,
                        data_dir=data_dir, train=train, classes='target', subclasses='targets', dualhead=True)

    clips = os.path.join(metadata_pth, '{}_clips_64frames.pt'.format('training' if train else 'testing'))
    clips = torch.load(clips)
    worked = list()
    failed = list()
    errors = list()
    for clip in tqdm.tqdm(clips, total=len(clips)):
        try:
            dataset.load_frames(clip)
            worked.append(clip)
        except Exception as e:
            print(e)
            errors.append(e)
            failed.append(clip)
    torch.save([failed, errors], 'datafiles/breakfast/testing_breakfast_dataset.pt')
    return worked, failed, errors


########################################################################################################################
# Data Pre-Processing Functions


@baker.command
def sample_clips(frames_dir, save_dir, split_pth='datafiles/breakfast/split.pt', clip_length=64):
    """
    This will first sample double the clip length uniformly, with a small overlap.
    Then it will subsample that every 2 frames.
    This reduces the total number of frames and hopes to cover more context.

    `save_compressed_preprocess_frames` was called first and took all frames however.
    It did a random horizontal crop and a center crop which does nothing because it was set at same H and W.

    :param str frames_dir: Where the compressed pickle files are for Breakfast video frames
    :param str save_dir:
    :param int clip_length: The desired end clip length
    :return:
    """
    split = torch.load(split_pth)
    for train, vids in enumerate([split[1], split[0]]):
        pths = [pth for pth in glob.glob(os.path.join(frames_dir, '*')) if pth.split('/')[-1].split('.')[0] in vids]
        data_list = list()
        for idx, pth in tqdm.tqdm(enumerate(pths), total=len(pths), desc='Training' if train else 'Testing'):
            vid = pth.split('/')[-1].split('.')[0]
            frames, targets, target = compress_pickle.load(pth)
            large_sample = video_utils.sample_frames(frames.shape[0], np.arange(frames.shape[0]), step=clip_length * 2,
                                                     window=2)
            for clip_num, sample in tqdm.tqdm(enumerate(large_sample), total=len(large_sample), position=1,
                                              leave=False):
                subsample = np.array([tmp[0] for tmp in video_utils.sample_frames(len(sample), sample, step=2,
                                                                                  window=1)])
                clip_num = "{:02d}".format(clip_num)
                save_pth = os.path.join(save_dir, vid + '_{}.gz'.format(clip_num))
                if not os.path.exists(save_pth):
                    compress_pickle.dump(frames[subsample], save_pth, compression='gzip')
                tmp = {'ce_steps': targets[subsample], 'frame_idces': subsample, 'ce_id': target, 'fps': 15,
                       'clip_n': clip_num, 'vid': vid, 'pth': save_pth}
                data_list.append(tmp)

        torch.save(data_list, os.path.join(save_dir, '{}_clips_{}frames.pth'.format('training' if train else 'testing',
                                                                                    clip_length)))
    print("Done")


@baker.command
def extract_asr_breakfast(key, save_audio_dir, save_asr_dir, video_dir='/home/c3-0/datasets/Breakfast/videos'):
    pths = [pth for pth in glob.glob(os.path.join(video_dir, '*/*/*')) if pth.endswith('.avi')]
    print("Converting .avi files to .wav files for Breakfast dataset...")
    for pth in tqdm.tqdm(pths, total=len(pths), desc='Converting MP4 to WAV'):
        identifier = pth.split('/')[-1].split('.')[0]
        save_audio_pth = os.path.join(save_audio_dir, identifier + '.wav')
        video_utils.convert_mp4_to_wav(pth, save_audio_pth)

    print("Extracting ASR from Breakfast audio files...")
    audio_utils.extract_asr(save_audio_dir, save_asr_dir, key, primary_language='en', secondary_language='es')

    print("Done")
    return


@baker.command
def generate_split_only(video_ids, video_dir='/home/c3-0/datasets/Breakfast/videos'):
    """
    Will generate a split file for training and testing based on person ID.
    It will also collect whichever ones fail because some frame extractions were interrupted.

    :param video_ids: The video IDs available for use.
    :param video_dir: The directory where the videos are stored.
    :return: [[Train Video Ids], [Validation Video Ids]]
    """
    failed = ['P45_webcam01_P45_salat', 'P07_webcam01_P07_cereals']
    person_names = os.listdir(video_dir)
    split_ratio = 0.85
    n_persons = len(person_names)
    n_persons_tr = int(n_persons * split_ratio)
    persons_tr = person_names[:n_persons_tr]

    train = list()
    test = list()
    for video_id in tqdm.tqdm(video_ids, total=len(video_ids), position=0):
        person = video_id.split('_')[0]
        if video_id not in failed:
            if person in persons_tr:
                train.append(video_id)
            else:
                test.append(video_id)
    torch.save([train, test], 'datafiles/breakfast/split.pt')


@baker.command
def generate_split_and_frame_mappings(videos_dir, frames_root_dir, unit_actions_pth, activities_pth, save_pth):
    """
    This is the frame mappings generator saved as a torch file.

    This will generate a split file of training video IDs and validation video IDs.
    It will also pre-process frames and sub-sample the desired amount, then saves as a compressed pickle file
        with gzip compression.

    The compressed files are saved in the directory passed as save_pth.

    :param videos_dir:
    :param frames_root_dir:
    :param unit_actions_pth:
    :param activities_pth:
    :param save_pth:
    :return:
    """
    print("Starting...")
    unit_actions_list = np.array(torch.load(unit_actions_pth) + ['walk_in', 'walk_out'])
    activities = torch.load(activities_pth)
    person_names = os.listdir(videos_dir)
    video_types = ['cam01', 'cam02', 'stereo', 'webcam01', 'webcam0']

    video_ids = list()
    for P in person_names:
        for vtype in video_types:
            pths = (glob.glob(os.path.join(videos_dir, P, vtype, '*.avi')))
            video_ids = np.concatenate([video_ids, ['_'.join(pth.split('/')[6:]).replace('.avi', '') for pth in pths]])

    # Generate Split
    split_ratio = 0.85
    n_persons = len(person_names)
    n_persons_tr = int(n_persons * split_ratio)
    persons_tr = person_names[:n_persons_tr]

    # Iterate through the videos and extract frames and target labeling
    train = dict()
    test = dict()
    failed = list()
    for video_id in tqdm.tqdm(video_ids, total=len(video_ids), position=0):
        person = video_id.split('_')[0]
        vid_type = video_id.split('_')[1]
        label_pth = os.path.join(videos_dir, person, vid_type, '_'.join(video_id.split('_')[2:]) + '.avi.labels')
        frames_root_path = os.path.join(frames_root_dir, video_id)
        video_frame_names = glob.glob(os.path.join(frames_root_path, '*'))

        # If frames not already extracted, skip
        if len(video_frame_names) < 1:
            failed.append(video_id)
            continue

        video_frame_names.sort()
        video_frame_names = np.array(video_frame_names)  # .astype(np.dtype('<U31'))

        frames = list()
        targets = list()

        # If labels are not available, skip
        if not os.path.exists(label_pth):
            failed.append(video_id)
            continue

        framewise_targets = open(label_pth, 'r')
        for line in tqdm.tqdm(framewise_targets.readlines(), position=1, leave=False):
            tgts = re.search('(?P<start>\d+)-(?P<end>\d+)\s(?P<cls>\w+)', line.replace('\n', ''))
            start = int(tgts.group("start")) - 1
            end = int(tgts.group("end")) - 1
            n_range = end - start
            if end > len(video_frame_names):
                end = len(video_frame_names) - 1

            frames = np.concatenate([frames, video_frame_names[np.arange(start, end)]])
            cls = np.argwhere(unit_actions_list == tgts.group('cls'))[0][0]

            targets = np.concatenate([targets, [cls] * n_range])

            activity = (set(video_id.split('_')) & set(activities)).pop()
            if video_id.split('_')[0] in persons_tr:
                train[video_id] = {'frames': frames, 'framewise_targets': targets, 'activity': activity}
            else:
                test[video_id] = {'frames': frames, 'framewise_targets': targets, 'activity': activity}
    print("Finished with {} train files and {} test files with {} failures".format(len(train.keys()), len(test.keys()),
                                                                                   len(failed)))
    torch.save([train, test], save_pth)


@baker.command
def save_compressed_preprocess_frames(frame_mappings, save_dir='datafiles/Breakfast/frames',
                                      use_dataloader=False):
    """
    Frames were already extracted so we combined them and compressed them using this function.
    :param frame_mappings:
    :param save_dir:
    :param use_dataloader:
    :return:
    """

    videos = torch.load(frame_mappings)
    for test, curr_videos in enumerate(videos):
        train = 'test' if test else 'train'

        # Some pre-processing in addition to normalization from `video_utils.LoadFrames`
        if train:
            transforms = Compose([video_utils.RandomCropVideo(224), video_utils.RandomHorizontalFlipVideo()])
        else:
            transforms = Compose(video_utils.CenterCropVideo(224))

        print("Starting for {} videos".format(len(curr_videos)))
        pbar = tqdm.tqdm(enumerate(curr_videos.items()), total=len(curr_videos))
        pbar.set_description(train)
        for idx, (video_id, video) in pbar:
            # In case their are concurrent programs running
            if video_id + '.gz' in os.listdir(save_dir):
                continue

            pths = video['frames']
            pths.sort()

            if use_dataloader:
                dataset = video_utils.LoadFrames(pths)
                dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=cpu_count(),
                                                         batch_size=cpu_count())
                frames = list()
                for idx, frame in enumerate(dataloader):
                    if idx == 0:
                        frames = frame
                    else:
                        frames = np.concatenate([frames, frame], axis=0)
            else:
                frames = np.array([video_utils.preprocess_img(frame) for frame in pths])

            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            frames = transforms(frames)

            targets = video['framewise_targets']
            activity = video['activity']
            compress_pickle.dump([frames, targets, activity],
                                 os.path.join(save_dir, '{}'.format(video_id)),
                                 compression='gzip')


if __name__ == '__main__':
    baker.run()
