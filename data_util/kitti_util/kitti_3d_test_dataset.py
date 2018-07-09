#/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import time

import cv2 as cv
import numpy as np
from PIL import Image
import copy
import time

from chainer import dataset
from data_util.kitti_util.input_velodyne import *
from data_util.kitti_util.cython_util.create_input import *


class Kitti3dTestDataset(dataset.DatasetMixin):
    def __init__(self, data_dir="./", split="train", ignore_labels=True,
                 crop_size=(713, 713), color_sigma=None, g_scale=[0.5, 2.0],
                 resolution=None, x_range=None, y_range=None, z_range=None,
                 l_rotate=None, g_rotate=None, voxel_shape=None,
                 t=35, thres_t=3, norm_input=False,
                 anchor_size=(1.56, 1.6, 3.9), anchor_center=(-1.0, 0., 0.),
                 fliplr=False, n_class=19, scale_label=1):
        split_dir = os.path.join(data_dir, 'ImageSets/')
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

        self.voxel_shape = voxel_shape
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.anchor_size = anchor_size
        self.anchor_center = anchor_center
        self.scale_label = scale_label
        self.t = t
        self.thres_t = thres_t
        self.norm_input = norm_input

        self.proj_cam = None
        self.calib = None

    def __len__(self):
        return len(self.velo_list)

    def get_example(self, i):
        pc = load_pointcloud_from_bin(self.velo_list[i])
        calib = read_calib_file(self.calib_list[i])
        proj_velo = proj_img_to_velo(calib) # R0_rect / Tr_velo_to_cam
        self.rect_cam = calib["P2"].reshape(3, 4)[:, :3]
        self.proj_cam = np.linalg.inv(proj_velo)

        random_indexes = np.random.permutation(pc.shape[0])
        pc = pc[random_indexes]

        d, h, w = self.voxel_shape
        d_res, h_res, w_res = self.resolution
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        rotate = 0

        pc = np.ascontiguousarray(pc, dtype=np.float32)
        create_input = create_feature_input_rotate if self.norm_input else create_feature_input
        s = time.time()
        feature_input, counter, indexes, n_no_empty = \
            create_input(pc, d_res, h_res, w_res, self.t,
                         d, h, w, x_min, x_max, y_min, y_max, z_min, z_max,
                         self.thres_t, 0, 92)
        print("create input", time.time() - s)

        area_mask = create_mask(0, 90, d, h, w, self.scale_label).astype(np.int8)

        return (feature_input, counter, indexes,
                np.array([indexes.shape[0]]), np.array([n_no_empty]), area_mask)
