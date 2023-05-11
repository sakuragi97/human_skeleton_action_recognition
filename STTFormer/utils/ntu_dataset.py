# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import os.path as osp
import pickle
import torchvision

import utils.augmentations as augmentations

from utils.augmentations import Normalize3D
from utils.base_dataset import BaseDataset


class NTUDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 augmentation=None,
                 split=None,
                 valid_ratio=None,
                 box_thr=0.5,
                 class_prob=None,
                 **kwargs):
        self.split = split

        super().__init__(
            ann_file, pipeline, augmentation, start_index=0, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            assert isinstance(self.valid_ratio, float)
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data


if __name__ == '__main__':
    ann_file = '/home/yas50454/datasets/NTU_Data/NTU_60/ntu60_3danno.pkl'
    normalizer = augmentations.Normalize3D()
    pipeline = torchvision.transforms.Compose([normalizer])

    noiser = augmentations.RandomAdditiveNoise(dist='NORMAL', prob=0.5, std=0.01)
    augmentation = torchvision.transforms.Compose([noiser])
    dataset = NTUDataset(ann_file, pipeline=None, split='xsub_train', num_classes=60, multi_class=True, augmentation=None)
    # dataset = NTUDataset(ann_file, pipeline=None, split='xsub_train')
    print(f"dataset size: {len(dataset)}")
    tmp = dataset[0]
    print(f"tmp['frame_dir']={tmp['frame_dir']}\ntmp['label']={tmp['label'].shape}\ntmp['keypoint']={tmp['keypoint'].shape}\ntmp['total_frames']={tmp['total_frames']}\ntmp['start_index']={tmp['start_index']}\ntmp['input']={tmp['input'].shape}\n")
    with open("/home/yas50454/datasets/NTU_Data/NTU_60/NTU_60_cross_subject_data_transform.pkl", 'wb') as f:
        pickle.dump(dataset, f)
        print(f"Pickle file saved!")
    print(1)
