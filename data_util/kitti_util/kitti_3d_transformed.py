#/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import time

import cv2 as cv
import numpy as np
from PIL import Image
import copy

from chainer import datasets
from chainercv import transforms
from data_util.kitti_util.kitti_3d_detection_dataset import \
    Kitti3dDetectionDataset
from data_util.kitti_util.input_velodyne import *
from data_util.kitti_util.cython_util.create_input import *
import chainer

def _transform(inputs, crop_size=(512, 512), g_scale=[0.95, 1.05],
               l_rotate=None, g_rotate=None, resolution=None, voxel_shape=None,
               x_range=None, y_range=None, z_range=None, t=35, thres_t=None,
               anchor_size=(1.56, 1.6, 3.9), anchor_center=(-1.0, 0., 0.),
               fliplr=False, n_class=20, scale_label=1, norm_input=False,
               label_rotate=False, angle=90, surround_prob=1):
    pc, places, rotates, size = inputs
    del inputs

    places[:, 2] += size[:, 0] / 2.

    rotates = np.pi / 2 - rotates
    rotates = aug_rotate(rotates, 0)

    # Local rotation (Label and local points) # TODO
    if l_rotate:
        l_rotate = np.random.uniform(l_rotate[0], l_rotate[1])

    # Global scaling
    if g_scale:
        scale = np.random.uniform(g_scale[0], g_scale[1], 4)
        pc *= scale
        places *= scale[:3]
        size *= scale[:3]

    # Global rotation
    r = 0
    if g_rotate:
        g_rotate = np.random.uniform(g_rotate[0], g_rotate[1])
        r = float(g_rotate / 180 * np.pi)
        rotate_matrix = np.array([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ])
        pc[:, :3] = np.dot(pc[:, :3], rotate_matrix.transpose())
        places = np.dot(places, rotate_matrix.transpose()).astype("f")
        rotates = aug_rotate(rotates, r)

    # Flip
    if fliplr:
        if np.random.rand() > 0.5:
            pc[:, 1] = pc[:, 1] * -1
            places[:, 1] = places[:, 1] * -1
            rotates = rotates * -1
            r = r * -1

    random_indexes = np.random.permutation(pc.shape[0])
    pc = pc[random_indexes]

    d, h, w = voxel_shape
    d_res, h_res, w_res = resolution
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    area_mask = create_mask(r, angle, d, h, w, scale_label)

    create_input = create_feature_input_rotate if norm_input else create_feature_input
    pc = np.ascontiguousarray(pc, dtype=np.float32)

    feature_input, counter, indexes, n_no_empty = \
        create_input(pc, d_res, h_res, w_res, t, d, h, w,
                     x_min, x_max, y_min, y_max, z_min, z_max, thres_t, r, angle)
    del pc

    anchor_z, anchor_y, anchor_x = anchor_center
    anchor_h, anchor_w, anchor_l = anchor_size

    # bird_corners = get_bird_boxcorners(places, rotates, size)

    label_creator = create_label_rotate if label_rotate else create_label
    gt_obj, gt_reg, gt_obj_for_reg = \
        label_creator(places, rotates, size, d_res, h_res, w_res, t, d, h, w,
                      x_min, x_max, y_min, y_max, z_min, z_max, thres_t,
                      anchor_l, anchor_w, anchor_h,
                      anchor_x, anchor_y, anchor_z, scale_label, surround_prob)
    return (feature_input, counter, indexes, gt_obj, gt_reg, gt_obj_for_reg,
             area_mask[None], np.array([indexes.shape[0]]), np.array([n_no_empty]))


class Kitti3dTransformedDataset(datasets.TransformDataset):
    def __init__(self, data_dir="./", split="train", ignore_labels=True,
                 crop_size=(713, 713), color_sigma=None, g_scale=[0.5, 2.0],
                 resolution=None, x_range=None, y_range=None, z_range=None,
                 l_rotate=None, g_rotate=None, voxel_shape=None,
                 t=35, thres_t=3, norm_input=False, label_rotate=False,
                 anchor_size=(1.56, 1.6, 3.9), anchor_center=(-1.0, 0., 0.),
                 fliplr=False, n_class=19, scale_label=1,
                 angle=90, surround_prob=0.8):
        self.d = Kitti3dDetectionDataset(
            data_dir, split, ignore_labels)
        t = partial(
            _transform, crop_size=crop_size, g_scale=g_scale,
            l_rotate=l_rotate, g_rotate=g_rotate, voxel_shape=voxel_shape,
            resolution=resolution, t=t, thres_t=thres_t, norm_input=norm_input,
            anchor_size=anchor_size, anchor_center=anchor_center,
            x_range=x_range, y_range=y_range, z_range=z_range,
            fliplr=fliplr, n_class=n_class, scale_label=scale_label,
            label_rotate=label_rotate, angle=angle, surround_prob=surround_prob)
        super().__init__(self.d, t)
