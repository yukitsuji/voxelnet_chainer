#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import cv2
import glob
import math
from data_util.kitti_util.parse_xml import parseXML

def read_label_from_xml(label_path):
    """Read label from xml file.

    # Returns:
        label_dic (dictionary): labels for one sequence.
        size (list): Bounding Box Size. [l, w. h]?
    """
    labels = parseXML(label_path)
    label_dic = {}
    for label in labels:
        first_frame = label.firstFrame
        nframes = label.nFrames
        size = label.size
        obj_type = label.objectType
        for index, place, rotate in zip(range(first_frame, first_frame+nframes),
                                        label.trans, label.rots):
            if index in label_dic.keys():
                label_dic[index]["place"] = np.vstack((label_dic[index]["place"], place))
                label_dic[index]["size"] = np.vstack((label_dic[index]["size"], np.array(size)))
                label_dic[index]["rotate"] = np.vstack((label_dic[index]["rotate"], rotate))
            else:
                label_dic[index] = {}
                label_dic[index]["place"] = place
                label_dic[index]["rotate"] = rotate
                label_dic[index]["size"] = np.array(size)
    return label_dic, size

def load_pointcloud_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def read_label_from_txt(label_path):
    """Read label from txt file."""
    text = np.fromfile(label_path)
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            label = label.split(" ")
            if label and label[0] == ("Car" or "Van"): #  or "Truck"
                bounding_box.append(label[8:15])

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6]
    else:
        return None, None, None

def read_calib_file(calib_path):
    """Read a calibration file."""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                print("Data Format is bad. Please confirm it")
    return data

def proj_img_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)

def read_labels(label_path, label_type="txt", is_velo_cam=True,
                proj_velo=None):
    """Read labels from xml or txt file.
       Original Label value is shifted about 0.27m from object center.
       So need to revise the position of objects.
    """
    if label_type == "txt":
        places, size, rotates = read_label_from_txt(label_path)
        if places is None:
            return None, None, None
        # rotates = np.pi / 2 - rotates
        if proj_velo is not None:
            places = np.dot(places, proj_velo.transpose())
        if is_velo_cam:
            places[:, 0] += 0.27

    elif label_type == "xml": # TODO
        bounding_boxes, size = read_label_from_xml(label_path)
        places = bounding_boxes[30]["place"]
        rotates = bounding_boxes[30]["rotate"][:, 2]
        size = bounding_boxes[30]["size"]

    return places.astype(np.float32), rotates.astype(np.float32), size.astype(np.float32)

def filter_camera_angle(places, angle=1.):
    """Filter pointclound by camera angle"""
    bool_in = np.logical_and((places[:, 1] * angle < places[:, 0]),
                             (-places[:, 1] * angle < places[:, 0]))
    return places[bool_in]

def get_boxcorners(places, rotates, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    for (place, rotate, sz) in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz

        corner = np.array([
            [- l / 2., - w / 2., 0],
            [- l / 2., + w / 2., 0],
            [+ l / 2., - w / 2., 0],
            [+ l / 2., + w / 2., 0],
            [- l / 2., - w / 2., + h],
            [- l / 2., + w / 2., + h],
            [+ l / 2., - w / 2., + h],
            [+ l / 2., + w / 2., + h],
        ])

        # corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        rotated_corner = np.dot(corner, rotate_matrix.transpose())
        rotated_corner += np.array([x, y, z])
        corners.append(rotated_corner)
    return np.array(corners)

def get_bird_boxcorners(places, rotates, size):
    """Create 4 corners of bounding box from bottom center."""
    corners = []
    for (place, rotate, sz) in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz

        corner = np.array([
            [- l / 2., - w / 2.],
            [- l / 2., + w / 2.],
            [+ l / 2., - w / 2.],
            [+ l / 2., + w / 2.],
        ])

        # corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate)],
            [np.sin(rotate), np.cos(rotate)],
        ])

        rotated_corner = np.dot(corner, rotate_matrix.transpose())
        rotated_corner += np.array([x, y])
        corners.append(rotated_corner)
    return np.array(corners, dtype="f")

def judge_in_voxel_area(points, x, y, z):
    x_logical = np.logical_and((points[:, 0] < x[1]), (points[:, 0] >= x[0]))
    y_logical = np.logical_and((points[:, 1] < y[1]), (points[:, 1] >= y[0]))
    z_logical = np.logical_and((points[:, 2] < z[1]), (points[:, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    return xyz_logical

def pointcloud_to_voxel(pc, resolution=0.50,
                        x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert pointCloud to Voxel"""
    xyz_logical = judge_in_voxel_area(pc, x, y, z)
    pc = pc[:, :3][xyz_logical]
    min_value = np.array([x[0], y[0], z[0]])
    pc = ((pc - min_value) / resolution).astype(np.int32)
    voxel = np.zeros((int((x[1] - x[0]) / resolution),
                      int((y[1] - y[0]) / resolution),
                      int(round((z[1]-z[0]) / resolution))))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    return voxel

def voxel_to_corner(center, corner):
    """Create 3D corner from voxel and the diff to corner"""
    return center + corner

def center_to_anchor(center, size, resolution=0.50, scale=4,
                     x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert original label to training label for objectness loss"""
    min_value = np.array([x[0], y[0], z[0]])
    xyz_logical = judge_in_voxel_area(center, x, y, z)
    center[:, 2] = center[:, 2] + size[:, 0] / 2. # Center of objects
    anchor_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    return anchor_center

def anchor_to_center(anchor_center, resolution=0.5, scale=4,
                     min_value=np.array([0., -50., -4.5])):
    """from sphere center to label center"""
    return anchor_center * (resolution * scale) + min_value

def corner_to_train(corners, anchor_center, resolution=0.50,
                    x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4):
    """Convert corner to training label for regression loss"""
    min_value = np.array([x[0], y[0], z[0]])
    x_logical = np.logical_and((corners[:, :, 0] < x[1]), (corners[:, :, 0] >= x[0]))
    y_logical = np.logical_and((corners[:, :, 1] < y[1]), (corners[:, :, 1] >= y[0]))
    z_logical = np.logical_and((corners[:, :, 2] < z[1]), (corners[:, :, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical)).all(axis=1)
    center = anchor_to_center(anchor_center, resolution=resolution,
                              scale=scale, min_value=min_value)
    train_corners = corners[xyz_logical] - center
    return train_corners

def create_training_label(center, size, corners, resolution=0.50, scale=4,
                 x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Create training labels which satisfy the range of experiment"""
    min_value = np.array([x[0], y[0], z[0]])
    xyz_logical = judge_in_voxel_area(center, x, y, z)
    center[:, 2] = center[:, 2] + size[:, 0] / 2. # Move bottom to center
    anchor_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    # from anchor to center. don't use original center information.
    center = anchor_to_center(anchor_center, resolution=resolution,
                              scale=scale, min_value=min_value)
    corners = corners[xyz_logical] - anchor_center[:, np.newaxis]
    return anchor_center, corners

def corner_to_voxel(voxel_shape, corners, anchor_center, scale=4):
    """Create final regression label from corner"""
    corner_voxel = np.zeros((voxel_shape[0] / scale, voxel_shape[1] / scale,
                             voxel_shape[2] / scale, 24)) # 24 is coordinate number
    corner_voxel[anchor_center[:, 0], anchor_center[:, 1], anchor_center[:, 2]] = corners
    return corner_voxel

def create_objectness_label(anchor_center, resolution=0.5,
                            x=90, y=100, z=10, scale=4):
    """Create Objectness label"""
    print(type(x[1] - x[0]))
    obj_maps = np.zeros((int((x[1] - x[0]) / (resolution * scale)),
                         int((y[1] - y[0]) / (resolution * scale)),
                         int((z[1] - z[0]) / (resolution * scale))))
    obj_maps[anchor_center[:, 0], anchor_center[:, 1], anchor_center[:, 2]] = 1
    return obj_maps

def process(velodyne_path, label_path=None, calib_path=None, dataformat="pcd",
            label_type="txt", is_velo_cam=False):
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None

    pc = load_pointcloud_from_bin(velodyne_path)

    if calib_path:
        calib = read_calib_file(calib_path)
        proj_velo = proj_img_to_velo(calib)

    if label_path:
        places, rotates, size = read_labels(label_path,
                                            label_type=label_type,
                                            is_velo_cam=is_velo_cam,
                                            proj_velo=proj_velo)

    corners = get_boxcorners(places, rotates, size)
    pc = filter_camera_angle(pc)
    print(pc.shape)
    anchor_center = center_to_anchor(places, size, resolution=0.25)
    bbox = anchor_to_center(anchor_center, resolution=0.25)

    import time
    s = time.time()
    voxel = pointcloud_to_voxel(pc, resolution=0.4)
    print("Voxelize", time.time() - s)
    # publish_pc2(pc, bbox.reshape(-1, 3))
    # publish_pc2(pc, corners.reshape(-1, 3))

def lidar_generator(batch_num, velodyne_path, label_path=None, calib_path=None,
                    resolution=0.2, dataformat="pcd", label_type="txt",
                    is_velo_cam=True, scale=4,
                    x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):
    velodynes_path = glob.glob(velodyne_path)
    labels_path = glob.glob(label_path)
    calibs_path = glob.glob(calib_path)
    velodynes_path.sort()
    labels_path.sort()
    calibs_path.sort()
    iter_num = len(velodynes_path) // batch_num

    for itn in range(iter_num):
        batch_voxel = []
        batch_g_map = []
        batch_g_cord = []

        for velodynes, labels, calibs in zip(velodynes_path[itn*batch_num:(itn+1)*batch_num],
                                             labels_path[itn*batch_num:(itn+1)*batch_num],
                                             calibs_path[itn*batch_num:(itn+1)*batch_num]):
            p = []
            pc = None
            bounding_boxes = None
            places = None
            rotates = None
            size = None
            proj_velo = None

            pc = load_pointcloud_from_bin(velodynes)

            if calib_path:
                calib = read_calib_file(calibs)
                proj_velo = proj_img_to_velo(calib)

            if label_path:
                places, rotates, size = read_labels(labels, label_type,
                                                    is_velo_cam=is_velo_cam,
                                                    proj_velo=proj_velo)
                if places is None:
                    continue

            corners = get_boxcorners(places, rotates, size)
            pc = filter_camera_angle(pc)

            voxel =  pointcloud_to_voxel(pc, resolution=resolution,
                                         x=x, y=y, z=z)
            center_anchor, corner_label = create_training_label(
                                              places, size, corners,
                                              resolution=resolution,
                                              x=x, y=y, z=z, scale=scale)

            if not center_anchor.shape[0]:
                continue
            g_map = create_objectness_label(center_anchor,
                                            resolution=resolution,
                                            x=(x[1] - x[0]),
                                            y=(y[1] - y[0]),
                                            z=(z[1] - z[0]),
                                            scale=scale)
            g_cord = corner_label.reshape(corner_label.shape[0], -1)
            g_cord = corner_to_voxel(voxel.shape, g_cord, center_anchor, scale=scale)

            batch_voxel.append(voxel)
            batch_g_map.append(g_map)
            batch_g_cord.append(g_cord)
        yield (np.array(batch_voxel, dtype=np.float32)[:, :, :, :, np.newaxis],
               np.array(batch_g_map, dtype=np.float32),
               np.array(batch_g_cord, dtype=np.float32))

if __name__ == "__main__":
    index = "003240"
    pcd_path = "/home/yukitsuji/dataset/kitti_detection/training/velodyne/{}.bin".format(index)
    label_path = "/home/yukitsuji/dataset/kitti_detection/training/label_2/{}.txt".format(index)
    calib_path = "/home/yukitsuji/dataset/kitti_detection/training/calib/{}.txt".format(index)
    process(pcd_path, label_path, calib_path=calib_path, dataformat="bin", is_velo_cam=True)
