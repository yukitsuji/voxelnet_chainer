#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import time
import os
import yaml

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training

subprocess.call(['sh', "setup.sh"])

from voxelnet.config_utils import *
from data_util.kitti_util.cython_util.create_input import create_anchor

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

from voxelnet.converter.voxelnet_concat import voxelnet_concat


def write_result(result_reg, result_prob, index, result_dir,
                 proj_cam, rect_cam):
    alpha = 0
    with open("{0}/{1:06d}.txt".format(result_dir, index), 'w') as f:
        result_reg[:, 0] -= 0.27
        result_reg[:, 2] -= result_reg[:, 5] / 2
        location = np.dot(result_reg[:, :3], proj_cam.transpose())
        for loc, reg, prob in zip(location, result_reg, result_prob):
            _, _, _, length, width, height, rotate = reg
            rotate = -rotate + np.pi / 2.
            x, y, z = loc
            bbox = [0, 0, 100, 100]
            f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                         "Car", 0, 0, alpha,
                         bbox[0], bbox[1], bbox[2], bbox[3],
                         height, width, length, x, y, z, rotate, prob))

def eval_voxelnet():
    """Test VoxelNet."""
    config, args = parse_args()
    model = get_model(config["model"])

    if args.gpu != -1:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    else:
        args.gpu = None

    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    total_iou, total_prob_loss, total_reg_loss = 0, 0, 0
    index = 0
    dataset_config = config['dataset']['test']['args']
    anchor_size = dataset_config['anchor_size']
    anchor_center = dataset_config['anchor_center']
    d, h, w = dataset_config['voxel_shape']
    d_res, h_res, w_res = dataset_config['resolution']
    x_min, x_max = dataset_config['x_range']
    y_min, y_max = dataset_config['y_range']
    z_min, z_max = dataset_config['z_range']
    scale_label = dataset_config['scale_label']
    anchor_z = anchor_center[0]
    anchor = create_anchor(d_res, h_res, w_res, d, h, w, anchor_z,
                           x_min, y_min, z_min, scale_label)

    result_dir = config['results']
    subprocess.call(['mkdir', '-p', result_dir])

    for batch in test_iter:
        s = time.time()
        batch = voxelnet_concat(batch, args.gpu)
        result_reg, result_prob = model.inference(*batch,
                                                  anchor=anchor,
                                                  anchor_center=anchor_center,
                                                  anchor_size=anchor_size,
                                                  thres_prob=args.thresh,
                                                  nms_thresh=args.nms_thresh)
        print("Time:", time.time() - s)
        write_result(result_reg, result_prob, index, result_dir,
                     test_iter.dataset.proj_cam, test_iter.dataset.rect_cam)
        index += 1
        print("########################################")
    print("##############  Statistic ##############")
    print("total iou", total_iou / index)
    print("prob loss", total_prob_loss / index)
    print("reg loss ", total_reg_loss / index)

def main():
    eval_voxelnet()

if __name__ == '__main__':
    main()
