#/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

import numpy as np

from chainer import dataset
from data_util.kitti_util.input_velodyne import *
from data_util.kitti_util.cython_util.create_input import *

class Kitti3dDetectionDataset(dataset.DatasetMixin):

    """Dataset class for a detection task on `Kitti 3D Dataset`_.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least three directories, :obj:`training`, `testing`
            and `ImageSets`.
        split ({'train', 'val'}): Select from dataset splits used in
            Cityscapes dataset.
    """

    def __init__(self, data_dir=None, split='train',
                 ignore_labels=True, **kwargs):
        split_dir = os.path.join(data_dir, 'ImageSets/')
        print(split)
        with open(split_dir + "{}.txt".format(split), 'r') as f:
            file_indexes = f.read().split('\n')

        base_dir = os.path.join(data_dir, 'training')
        velo_dir = os.path.join(base_dir, 'velodyne/')
        label_dir = os.path.join(base_dir, 'label_2/')
        calib_dir = os.path.join(base_dir, 'calib/')

        if not file_indexes[-1]:
            file_indexes = file_indexes[:-1]

        self.velo_list = [velo_dir + index + '.bin' for index in file_indexes]
        self.label_list = [label_dir + index + '.txt' for index in file_indexes]
        self.calib_list = [calib_dir + index + '.txt' for index in file_indexes]
        self.ignore_labels = ignore_labels

        print(len(self.velo_list))

    def __len__(self):
        return len(self.velo_list)

    def get_example(self, i):
        """

        Args:
            i (int): The index of the example.

        Returns:
            tuple of a color image and a label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the image.
            The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        while True:
            pc = load_pointcloud_from_bin(self.velo_list[i])
            calib = read_calib_file(self.calib_list[i])
            proj_velo = proj_img_to_velo(calib)
            places, rotates, size = read_labels(self.label_list[i],
                                                label_type='txt',
                                                is_velo_cam=True,
                                                proj_velo=proj_velo)
            if places is None:
                i += 1
            else:
                break
        return pc, places, rotates, size
