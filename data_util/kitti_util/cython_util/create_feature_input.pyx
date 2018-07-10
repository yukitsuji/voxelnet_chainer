# distutils: language=c++
# -*- coding: utf-8 -*-

cimport cython
import numpy as np
cimport numpy as np

import gc
import sys

from libc.math cimport log
from libc.math cimport sin, cos
from libc.math cimport abs as c_abs
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t

cdef inline DTYPE_t max_float(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline DTYPE_t min_float(np.float32_t a, np.float32_t b):
    return a if a <= b else b

cdef inline int max_int(int a, int b):
    return a if a >= b else b

cdef inline int min_int(int a, int b):
    return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def create_anchor(
        DTYPE_t D_res, DTYPE_t H_res, DTYPE_t W_res,
        float D, float H, float W, float anchor_z,
        float x_min, float y_min, float z_min, float scale_label):
    cdef int anchor_h = int(H / scale_label)
    cdef int anchor_w = int(W / scale_label)
    cdef float anchor_x, anchor_y
    cdef np.ndarray[DTYPE_t, ndim=3] anchor = np.zeros((anchor_h, anchor_w, 3), dtype=DTYPE)
    cdef int i, j
    cdef float H_res_scale = H_res * scale_label
    cdef float W_res_scale = W_res * scale_label

    for i in range(anchor_h):
      anchor_y = i * H_res_scale + y_min
      for j in range(anchor_w):
        anchor_x = j * W_res_scale + x_min
        anchor[i, j, 0] = anchor_x
        anchor[i, j, 1] = anchor_y
        anchor[i, j, 2] = anchor_z
    return anchor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
cdef int is_inside_area(DTYPE_t x, DTYPE_t y,
                        float left_ratio, float right_ratio):
    cdef float ratio
    ratio = left_ratio if y > 0 else right_ratio
    return 1 if c_abs(x / (y + 1e-5)) > ratio else 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def create_feature_input_rotate(
        np.ndarray[DTYPE_t, ndim=2, mode='c'] points,
        DTYPE_t D_res, DTYPE_t H_res, DTYPE_t W_res,
        int T, int D, int H, int W,
        float x_min, float x_max, float y_min, float y_max,
        float z_min, float z_max, int thres_t, float rotate, float angle):
    """
    Parameters
    ----------
    points: (Num of points, 4) ndarray of float32_t
    D_res:
    H_res:
    W_res:
    T: int. Maximum number of non-empty voxel
    D: int. Depth of voxel
    H: int. Height of voxel
    W: int. Width of voxel
    Returns
    -------
    feature: (DHW, 4) ndarray of float32_t
    indexes: (DHW) ndarray of int32_t
    """
    cdef int num_points = points.shape[0]
    cdef int HW = H * W
    cdef int DHW = D * H * W
    cdef np.ndarray[DTYPE_t, ndim=3] feature = np.zeros((DHW / 100, 7, T), dtype=DTYPE)
    cdef np.ndarray[DTYPE_int_t, ndim=1] counter = np.zeros((DHW / 100), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=1] indexes = np.zeros((DHW / 100), dtype=DTYPE_int)
    cdef int n, count, c, n_no_empty, i, j
    cdef int d_data, h_data, w_data
    cdef DTYPE_t x, y, z, reflect
    cdef int index
    cdef float mean_x, mean_y, mean_z
    cdef float x_min_1 = x_min + 0.01
    cdef float y_min_1 = y_min + 0.01
    cdef float z_min_1 = z_min + 0.01
    cdef float x_max_1 = x_max - 0.01
    cdef float y_max_1 = y_max - 0.01
    cdef float z_max_1 = z_max - 0.01
    cdef float x_range = x_max - x_min
    cdef float y_range = y_max - y_min
    cdef float z_range = z_max - z_min

    cdef float left_ratio, right_ratio, base_angle

    base_angle = (angle / 2) / 180 * float(np.pi)
    left_ratio = c_abs(cos(base_angle + rotate) / sin(base_angle + rotate))
    right_ratio = c_abs(cos(base_angle - rotate) / sin(base_angle - rotate))

    cdef unordered_map[int, int] hash_table = unordered_map[int, int]()
    cdef unordered_map[int, int].iterator iterator
    cdef int T_int = <int>T
    n_no_empty = 0
    for n in range(num_points):
      x = points[n, 0]
      y = points[n, 1]
      if x > x_min_1 and x < x_max_1 and y > y_min_1 and y < y_max_1:
        if is_inside_area(x, y, left_ratio, right_ratio):
          z = points[n, 2]
          if z > z_min_1 and z < z_max_1:
            d_data = int((z - z_min) / D_res)
            h_data = int((y - y_min) / H_res)
            w_data = int((x - x_min) / W_res)
            index = d_data * HW + h_data * W + w_data
            iterator = hash_table.find(index)
            if (iterator == hash_table.end()):
                hash_table[index] = n_no_empty
                indexes[n_no_empty] = index
                index = n_no_empty
                n_no_empty += 1
            else:
                index = hash_table[index]
            count = counter[index]
            if count < T_int:
              feature[index, 0, count] = x
              feature[index, 1, count] = y
              feature[index, 2, count] = z
              feature[index, 3, count] = points[n, 3]
              counter[index] = count + 1

    mean_x = 0
    mean_y = 0
    mean_z = 0
    for n in range(n_no_empty):
      count = <int>counter[n]
      if count > thres_t:
        mean_x = 0
        mean_y = 0
        mean_z = 0
        for c in range(count):
          mean_x += feature[n, 0, c]
          mean_y += feature[n, 1, c]
          mean_z += feature[n, 2, c]

        mean_x = mean_x / count
        mean_y = mean_y / count
        mean_z = mean_z / count

        for c in range(count):
          feature[n, 4, c] = feature[n, 0, c] - mean_x
          feature[n, 5, c] = feature[n, 1, c] - mean_y
          feature[n, 6, c] = feature[n, 2, c] - mean_z
      else:
        for c in range(count):
          feature[n, 0, c] = 0
          feature[n, 1, c] = 0
          feature[n, 2, c] = 0
          feature[n, 3, c] = 0
    return feature, counter, indexes, n_no_empty

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def create_feature_input(
        np.ndarray[DTYPE_t, ndim=2, mode='c'] points,
        DTYPE_t D_res, DTYPE_t H_res, DTYPE_t W_res,
        int T, int D, int H, int W,
        float x_min, float x_max, float y_min, float y_max,
        float z_min, float z_max, int thres_t, float rotate, float angle):
    """
    Parameters
    ----------
    points: (Num of points, 4) ndarray of float32_t
    D_res:
    H_res:
    W_res:
    T: int. Maximum number of non-empty voxel
    D: int. Depth of voxel
    H: int. Height of voxel
    W: int. Width of voxel
    Returns
    -------
    feature: (DHW, 4) ndarray of float32_t
    indexes: (DHW) ndarray of int32_t
    """
    cdef int num_points = points.shape[0]
    cdef int HW = H * W
    cdef int DHW = D * H * W
    cdef np.ndarray[DTYPE_t, ndim=3] feature = np.zeros((DHW / 100, 7, T), dtype=DTYPE)
    cdef np.ndarray[DTYPE_int_t, ndim=1] counter = np.zeros((DHW / 100), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=1] indexes = np.zeros((DHW / 100), dtype=DTYPE_int)
    cdef int n, count, c, n_no_empty, i, j
    cdef int d_data, h_data, w_data
    cdef DTYPE_t x, y, z, reflect
    cdef int index
    cdef float mean_x, mean_y, mean_z
    cdef float x_min_1 = x_min + 0.01
    cdef float y_min_1 = y_min + 0.01
    cdef float z_min_1 = z_min + 0.01
    cdef float x_max_1 = x_max - 0.01
    cdef float y_max_1 = y_max - 0.01
    cdef float z_max_1 = z_max - 0.01
    cdef float x_range = x_max - x_min
    cdef float y_range = y_max - y_min
    cdef float z_range = z_max - z_min

    cdef unordered_map[int, int] hash_table = unordered_map[int, int]()
    cdef unordered_map[int, int].iterator iterator
    cdef int T_int = <int>T
    n_no_empty = 0
    for n in range(num_points):
      x = points[n, 0]
      y = points[n, 1]
      if x > x_min_1 and x < x_max_1 and y > y_min_1 and y < y_max_1:
        if x > c_abs(y):
          z = points[n, 2]
          if z > z_min_1 and z < z_max_1:
            d_data = int((z - z_min) / D_res)
            h_data = int((y - y_min) / H_res)
            w_data = int((x - x_min) / W_res)
            index = d_data * HW + h_data * W + w_data
            iterator = hash_table.find(index)
            if (iterator == hash_table.end()):
                hash_table[index] = n_no_empty
                indexes[n_no_empty] = index
                index = n_no_empty
                n_no_empty += 1
            else:
                index = hash_table[index]
            count = counter[index]
            if count < T_int:
              feature[index, 0, count] = x
              feature[index, 1, count] = y
              feature[index, 2, count] = z
              feature[index, 3, count] = points[n, 3]
              counter[index] = count + 1
    mean_x = 0
    mean_y = 0
    mean_z = 0
    for n in range(n_no_empty):
      count = <int>counter[n]
      if count > thres_t:
        mean_x = 0
        mean_y = 0
        mean_z = 0
        for c in range(count):
          mean_x += feature[n, 0, c]
          mean_y += feature[n, 1, c]
          mean_z += feature[n, 2, c]

        mean_x = mean_x / count
        mean_y = mean_y / count
        mean_z = mean_z / count

        for c in range(count):
          feature[n, 4, c] = feature[n, 0, c] - mean_x
          feature[n, 5, c] = feature[n, 1, c] - mean_y
          feature[n, 6, c] = feature[n, 2, c] - mean_z
    return feature, counter, indexes, n_no_empty

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def create_label_rotate(
        np.ndarray[DTYPE_t, ndim=2] places,
        np.ndarray[DTYPE_t, ndim=1] rotates,
        np.ndarray[DTYPE_t, ndim=2] size,
        DTYPE_t D_res, DTYPE_t H_res, DTYPE_t W_res,
        int T, int D, int H, int W,
        float x_min, float x_max, float y_min, float y_max,
        float z_min, float z_max, int thres_t,
        float anchor_l, float anchor_w, float anchor_h,
        float anchor_x, float anchor_y, float anchor_z,
        int scale_label, float surround_prob):
    """
    Parameters
    ----------
    points: (Num of points, 4) ndarray of float32_t
    D_res:
    H_res:
    W_res:
    T: int. Maximum number of non-empty voxel
    D: int. Depth of voxel
    H: int. Height of voxel
    W: int. Width of voxel
    Returns
    -------
    feature: (DHW, 4) ndarray of float32_t
    indexes: (DHW) ndarray of int32_t
    """
    cdef int num_labels = places.shape[0]
    cdef int anchor_H = int(H / scale_label)
    cdef int anchor_W = int(W / scale_label)

    cdef np.ndarray[DTYPE_int_t, ndim=2] gt_obj = np.zeros((anchor_H, anchor_W), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_t, ndim=3] gt_reg = np.zeros((8, anchor_H, anchor_W), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] gt_obj_for_reg = np.zeros((anchor_H, anchor_W), dtype=DTYPE)

    cdef int n, count, c, n_no_empty, i, j, x_label, y_label, a, b
    cdef float height, width, length, rotate, length_2, width_2
    cdef int x_label_b, y_label_a
    cdef float anchor_x_b, anchor_y_a
    cdef DTYPE_t x, y, z, reflect

    cdef float x_min_1 = x_min + 0.0011
    cdef float y_min_1 = y_min + 0.0011
    cdef float z_min_1 = z_min + 0.0011
    cdef float x_max_1 = x_max - 0.0011
    cdef float y_max_1 = y_max - 0.0011
    cdef float z_max_1 = z_max - 0.0011

    cdef float W_res_scale = W_res * scale_label
    cdef float H_res_scale = H_res * scale_label
    cdef float pi = float(np.pi)
    cdef float pi_2 = pi / 2.
    cdef int pi_true

    for n in range(num_labels):
      x = places[n, 0]
      y = places[n, 1]
      z = places[n, 2]

      if x > x_min_1 and x < x_max_1 and y > y_min_1 and y < y_max_1 and z > z_min_1 and z < z_max_1:
        height = size[n, 0]
        width = size[n, 1]
        length = size[n, 2]
        rotate = rotates[n]

        x_label = <int>((x - x_min) / (W_res * scale_label))
        y_label = <int>((y - y_min) / (H_res * scale_label))

        anchor_x = x_label * W_res_scale + x_min
        anchor_y = y_label * H_res_scale + y_min

        gt_obj[y_label, x_label] = 1
        pi_true = 1
        if rotate < 0:
          rotate = pi + rotate
        if rotate >= pi_2:
          pi_true = 0
          rotate = pi - rotate

        for i in range(-1, 2):
          for j in range(-1, 2):
            x_label_b = x_label + j
            y_label_a = y_label + i
            if (y_label_a) >= 0 and (y_label_a) < anchor_H:
              if (x_label_b) >= 0 and (x_label_b) < anchor_W:
                gt_obj_for_reg[y_label_a, x_label_b] = surround_prob
                anchor_x_b = anchor_x + j * W_res_scale
                anchor_y_a = anchor_y + i * H_res_scale
                gt_reg[0, y_label_a, x_label_b] = (x - anchor_x_b) / anchor_l
                gt_reg[1, y_label_a, x_label_b] = (y - anchor_y_a) / anchor_w
                gt_reg[2, y_label_a, x_label_b] = (z - anchor_z) / anchor_h
                gt_reg[3, y_label_a, x_label_b] = log(length / anchor_l)
                gt_reg[4, y_label_a, x_label_b] = log(width / anchor_w)
                gt_reg[5, y_label_a, x_label_b] = log(height / anchor_h)
                gt_reg[6, y_label_a, x_label_b] = rotate / pi_2
                gt_reg[7, y_label_a, x_label_b] = pi_true
        gt_obj_for_reg[y_label, x_label] = 1
    return gt_obj, gt_reg, gt_obj_for_reg

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def create_label(
        np.ndarray[DTYPE_t, ndim=2] places,
        np.ndarray[DTYPE_t, ndim=1] rotates,
        np.ndarray[DTYPE_t, ndim=2] size,
        DTYPE_t D_res, DTYPE_t H_res, DTYPE_t W_res,
        int T, int D, int H, int W,
        float x_min, float x_max, float y_min, float y_max,
        float z_min, float z_max, int thres_t,
        float anchor_l, float anchor_w, float anchor_h,
        float anchor_x, float anchor_y, float anchor_z,
        int scale_label, float surround_prob):
    """
    Parameters
    ----------
    points: (Num of points, 4) ndarray of float32_t
    D_res:
    H_res:
    W_res:
    T: int. Maximum number of non-empty voxel
    D: int. Depth of voxel
    H: int. Height of voxel
    W: int. Width of voxel
    Returns
    -------
    feature: (DHW, 4) ndarray of float32_t
    indexes: (DHW) ndarray of int32_t
    """
    cdef int num_labels = places.shape[0]
    cdef int anchor_H = int(H / scale_label)
    cdef int anchor_W = int(W / scale_label)

    cdef np.ndarray[DTYPE_int_t, ndim=2] gt_obj = np.zeros((anchor_H, anchor_W), dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_t, ndim=3] gt_reg = np.zeros((7, anchor_H, anchor_W), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] gt_obj_for_reg = np.zeros((anchor_H, anchor_W), dtype=DTYPE)

    cdef int n, count, c, n_no_empty, i, j, x_label, y_label, a, b
    cdef float height, width, length, rotate, length_2, width_2
    cdef int x_label_b, y_label_a
    cdef float anchor_x_b, anchor_y_a
    cdef DTYPE_t x, y, z, reflect
    cdef float x_min_1 = x_min + 0.0011
    cdef float y_min_1 = y_min + 0.0011
    cdef float z_min_1 = z_min + 0.0011
    cdef float x_max_1 = x_max - 0.0011
    cdef float y_max_1 = y_max - 0.0011
    cdef float z_max_1 = z_max - 0.0011

    cdef float W_res_scale = W_res * scale_label
    cdef float H_res_scale = H_res * scale_label

    for n in range(num_labels):
      x = places[n, 0]
      y = places[n, 1]
      z = places[n, 2]

      if x > x_min_1 and x < x_max_1 and y > y_min_1 and y < y_max_1 and z > z_min_1 and z < z_max_1:
        height = size[n, 0]
        width = size[n, 1]
        length = size[n, 2]
        rotate = rotates[n]

        x_label = <int>((x - x_min) / (W_res * scale_label))
        y_label = <int>((y - y_min) / (H_res * scale_label))

        anchor_x = x_label * W_res_scale + x_min
        anchor_y = y_label * H_res_scale + y_min

        gt_obj[y_label, x_label] = 1
        if rotate < 0:
          rotate = 3.14160 + rotate
        for i in range(-1, 2):
          for j in range(-1, 2):
            x_label_b = x_label + j
            y_label_a = y_label + i
            if (y_label_a) >= 0 and (y_label_a) < anchor_H:
              if (x_label_b) >= 0 and (x_label_b) < anchor_W:
                gt_obj_for_reg[y_label_a, x_label_b] = surround_prob
                anchor_x_b = anchor_x + j * W_res_scale
                anchor_y_a = anchor_y + i * H_res_scale
                gt_reg[0, y_label_a, x_label_b] = (x - anchor_x_b) / anchor_l
                gt_reg[1, y_label_a, x_label_b] = (y - anchor_y_a) / anchor_w
                gt_reg[2, y_label_a, x_label_b] = (z - anchor_z) / anchor_h
                gt_reg[3, y_label_a, x_label_b] = log(length / anchor_l)
                gt_reg[4, y_label_a, x_label_b] = log(width / anchor_w)
                gt_reg[5, y_label_a, x_label_b] = log(height / anchor_h)
                gt_reg[6, y_label_a, x_label_b] = rotate / 3.14160
        gt_obj_for_reg[y_label, x_label] = 1
    return gt_obj, gt_reg, gt_obj_for_reg

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def aug_rotate(
        np.ndarray[DTYPE_t, ndim=1] rotates,
        float r):
    """Data augmentation for global ratation.

    Args:
        rotates(ndarray): range is [-pi, pi]
        r(float): range is [-pi, pi]
    """
    cdef int num_labels = rotates.shape[0]
    cdef float rotate
    cdef int n
    cdef float pi = float(np.pi)

    for n in range(num_labels):
      rotate = rotates[n]
      rotate += r
      if rotate >= pi:
        rotate = rotate - 2 * pi
      elif rotate <= -pi:
        rotate = 2 * pi + rotate

      rotates[n] = rotate
    return rotates

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def create_mask(float r, float angle,
                int D, int H, int W, int scale_label):
    """Confidence mask. If the area is in the range of detection.

    Args:
        r(float): range is [-pi, pi]
    """
    cdef int anchor_H = int(H / scale_label)
    cdef int anchor_W = int(W / scale_label)
    cdef np.ndarray[DTYPE_int_t, ndim=2] mask = np.zeros((anchor_H, anchor_W),
                                                         dtype=DTYPE_int)
    cdef float base_angle = (angle / 2) / 180 * float(np.pi)
    cdef float left_ratio = c_abs(cos(base_angle + r) / sin(base_angle + r))
    cdef float right_ratio = c_abs(cos(base_angle - r) / sin(base_angle - r))
    cdef float half_h = (anchor_H - 1) / 2.
    cdef float y

    for h in range(anchor_H):
        for w in range(anchor_W):
            y = h - half_h + 1e-5
            ratio = left_ratio if y >= 0 else right_ratio
            if c_abs(w / y) > ratio:
              mask[h, w] = 1
    return mask
