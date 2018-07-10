#!/usr/bin/env python
import argparse
import sys
import os
import numpy as np
import cv2
import glob
import math
import yaml
import subprocess

try:
    from voxelnet.config_utils import *
    from voxelnet.models.feature_to_voxel import feature_to_voxel
    import chainer.functions as F
except:
    pass

import matplotlib.pyplot as plt

try:
    import rospy
    from data_util.kitti_util.input_velodyne import *
    from data_util.kitti_util.parse_xml import parseXML
    import std_msgs.msg
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2
except:
    pass

def publish_pc2(pc, obj, pre_obj=None, pc_1=None):
    """Publisher of PointCloud data"""
    pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=10)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])

    pub2 = rospy.Publisher("/points_raw1", PointCloud2, queue_size=10)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points2 = pc2.create_cloud_xyz32(header, obj)

    if pre_obj is not None:
        pub3 = rospy.Publisher("/points_raw2", PointCloud2, queue_size=10)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne"
        points3 = pc2.create_cloud_xyz32(header, pre_obj)

    if pc_1 is not None:
        pub4 = rospy.Publisher("/points_raw3", PointCloud2, queue_size=10)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "velodyne"
        points4 = pc2.create_cloud_xyz32(header, pc_1[:, :3])

    r = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        pub.publish(points)
        pub2.publish(points2)
        if pre_obj is not None:
            pub3.publish(points3)
        if pc_1 is not None:
            pub4.publish(points4)
        r.sleep()

def viz_rviz(index, base_path, dataformat="bin",
             label_type="txt", is_velo_cam=True, predicted_path=None):

    velodyne_path = "{}/velodyne/{}.bin".format(base_path, index)
    label_path = "{}/label_2/{}.txt".format(base_path, index)
    calib_path = "{}/calib/{}.txt".format(base_path, index)

    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None

    if dataformat == "bin":
        pc = load_pointcloud_from_bin(velodyne_path)

    if calib_path:
        calib = read_calib_file(calib_path)
        proj_velo = proj_img_to_velo(calib)[:, :3]

    r = 0#-np.pi / 12
    rotate_matrix = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r), np.cos(r), 0],
        [0, 0, 1]
    ])
    # pc[:, :3] = np.dot(pc[:, :3], rotate_matrix.transpose())
    # pc[:, 1] *= -1

    if label_path:
        places, rotates, size = read_labels(label_path, label_type,
                                            is_velo_cam=is_velo_cam,
                                            proj_velo=proj_velo)
        print(places)
        # print(rotates)
        # rotates[:] = 0
        places = np.dot(places, rotate_matrix.transpose())
        rotates = np.pi / 2 - rotates
        # rotates += r

        # pi = np.pi
        # for index, i in enumerate(rotates):
        #     if i >= pi:
        #         rotates[index] = i - 2 * pi
        #     elif i <= -pi:
        #         rotates[index] = 2 * pi + i
        # rotates *= -1
        # places[:, 1] *= -1
        #
        # print(rotates)
        #
        # for index, i in enumerate(rotates):
        #     if i < 0:
        #       rotates[index] = pi + i
        #     if i >= pi/2:
        #       rotates[index] = -(pi - i)
        #
        # print(rotates)
        #
        # rotates = np.pi / 2 - rotates

        # rotates = - rotates + np.pi / 2
        # rotates = np.pi / 2 - rotates

        corners = get_boxcorners(places, rotates, size)
        corners = corners.reshape(-1, 3)

    pre_corners = None
    if predicted_path is not None:
        predicted_label_path = "{}/{}.txt".format(predicted_path, index)
        pre_places, pre_rotates, pre_size = read_labels(predicted_label_path,
                                                        label_type,
                                                        is_velo_cam=is_velo_cam,
                                                        proj_velo=proj_velo)
        print(pre_places)
        pre_rotates = np.pi / 2 - pre_rotates
        pre_corners = get_boxcorners(pre_places, pre_rotates, pre_size)
        pre_corners = pre_corners.reshape(-1, 3)

    pc = filter_camera_angle(pc)
    print("mean", pc.mean(axis=0))
    anchor_center = center_to_anchor(places, size, resolution=0.25)
    bbox = anchor_to_center(anchor_center, resolution=0.25)

    # hoge = np.zeros((19, 400, 400, 2), dtype='f')
    hoge = np.zeros((19, 800, 800, 2), dtype='f')
    pc = pc[pc[:, 2] < 1.0]
    pc = pc[pc[:, 2] > -2.8]

    x_min = 0
    y_min = -40
    z_min = -2.8
    for index, p in enumerate(pc):
        x, y, z, intensity = p
        intensity += 1.0
        if x <= 0 or x >= 70.4 or y <= -40 or y >= 40 or z <= -2.8 or z >= 1.2:
            continue
        x -= x_min
        y -= y_min
        z -= z_min
        x /= 0.1
        y /= 0.1
        z /= 0.2
        if hoge[int(z), int(y), int(x), 0] < intensity:
            hoge[int(z), int(y), int(x), 0] = intensity
            hoge[int(z), int(y), int(x), 1] = index
    index = hoge[:, :, :, 1]
    index = index[index != 0].astype('i')
    b_pc = np.zeros((pc.shape[0],), dtype='bool')
    b_pc[index] = True
    # pc = pc[~b_pc]
    print("OK", len(index))
    # publish_pc2(pc, bbox.reshape(-1, 3))
    publish_pc2(pc[b_pc], corners, pre_obj=pre_corners, pc_1=pc[~b_pc])

def viz_input(args, config):
    """Visualize input for network."""
    subprocess.call(['sh', "setup.sh"])
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    dataset_config = config['dataset']['test']['args']
    for batch in test_iter:
        x_list, counter, indexes_list, gt_prob, gt_reg, batch, n_no_empty = batch[0]
        gt_prob = F.resize_images(gt_prob.astype("f")[np.newaxis, np.newaxis],
                                  (400, 352))[0, 0].data
        len_image = len(x_list)
        fig, axes = plt.subplots(2, 3, figsize=(20, 7))
        thres_list = dataset_config['thres_t']
        for index, (x, indexes) in enumerate(zip(x_list, indexes_list)):
            x = x[:, :, 0]
            x = feature_to_voxel(x, indexes, 3, 10, 400, 352, batch)
            input_x = chainer.cuda.to_cpu(x.data.astype("f")[0])
            input_x = input_x.max(axis=(0, 1))
            image = np.ones(input_x.shape, dtype='f') * 0.95
            slice1 = image.copy()
            slice2 = image.copy()
            slice3 = image.copy()
            slice1[input_x != 0] = 0.3
            slice2[input_x != 0] = .8
            slice3[input_x != 0] = 0.2
            slice1[gt_prob != 0] = 1
            slice2[gt_prob != 0] = 0
            slice3[gt_prob != 0] = 0
            image = np.ones((400, 352, 3))
            image[:, :, 0] = slice1
            image[:, :, 1] = slice2
            image[:, :, 2] = slice3
            i = int(index / 3)
            j = int(index % 3)
            axes[i, j].imshow(image[::-1][100:300], cmap="hot")
            axes[i, j].axis('off')
            axes[i, j].set_title("Thres: {}".format(thres_list[index]))
        plt.tight_layout()
        plt.show()

def filter_camera_angle(places, angle=1.):
    """Filter pointclound by camera angle"""
    bool_in = np.logical_and((places[:, 1] * angle < places[:, 0]),
                             (-places[:, 1] * angle < places[:, 0]))
    return places[bool_in]

def calc_stats(args, config):
    """Calculate statistic of raw data"""
    subprocess.call(['sh', "setup.sh"])
    model = get_model(config["model"])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    dataset_config = config['dataset']['test']['args']
    len_dataset = len(test_iter.dataset)
    sum_hist = None
    for i, batch in enumerate(test_iter):
        print(i)
        pc, places, rotates, size = batch[0]
        pc = filter_camera_angle(pc)
        hist = np.histogram(pc[:, 1], bins=8, range=(-40, 40))
        if sum_hist is None:
            sum_hist = hist[0] / len_dataset
        else:
            sum_hist += hist[0] / len_dataset
    plt.bar(hist[1][:8] + 5, sum_hist, width=10)
    plt.title("Raw data of velodyne")
    plt.xlabel("y(m)", fontsize=12)
    plt.ylabel("frequency", fontsize=12)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='rviz', type=str,
                        help='Decide process')
    parser.add_argument('--config', default=None, type=str,
                        help='configuration file')
    parser.add_argument('--base_path',
                        default="/home/yukitsuji/dataset/kitti_detection/training",
                        type=str, help='image path')
    parser.add_argument('--predicted_path',
                         type=str, help="predicted label dir")
    parser.add_argument('--index', default="003241", type=str,
                        help='index of image')

    args = parser.parse_args()
    config = yaml.load(open(args.config)) if args.config else None
    return args, config

if __name__ == "__main__":
    args, config = parse_args()
    if args.type == 'rviz':
        index = args.index
        base_path = args.base_path
        viz_rviz(index, base_path, dataformat="bin",
                 is_velo_cam=True, predicted_path=args.predicted_path)
    elif args.type == 'input':
        viz_input(args, config)
    elif args.type == 'stats':
        calc_stats(args, config)
