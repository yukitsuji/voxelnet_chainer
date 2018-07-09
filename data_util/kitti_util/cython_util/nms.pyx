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
def nms_3d(np.ndarray[DTYPE_t, ndim=2] pred_reg,
           np.ndarray[DTYPE_t, ndim=1] pred_prob,
           float threshold):
    cdef int num_candidate = pred_prob.shape[0]
    cdef np.ndarray[DTYPE_int_t, ndim=1] suppressed = np.zeros((num_candidate), dtype=DTYPE_int)
    cdef int i, j

    cdef float center_x, center_y, length, width
    cdef float xx1, xx2, yy1, yy2

    cdef float com_center_x, com_center_y, com_length, com_width
    cdef float com_xx1, com_xx2, com_yy1, com_yy2
    cdef float rotate, rotate_abs

    cdef left_x, right_x, top_y, bottom_y

    result_index = []
    cdef float pi = float(np.pi)
    for i in range(num_candidate):
      if suppressed[i] == 1:
        continue
      result_index.append(i)
      center_x = pred_reg[i, 0]
      center_y = pred_reg[i, 1]
      rotate = pred_reg[i, 6]
      rotate_abs = c_abs(rotate % pi)
      if rotate_abs > (pi / 4) and rotate_abs < (pi * 3 / 4):
        length = pred_reg[i, 3]
        width = pred_reg[i, 4]
      else:
        length = pred_reg[i, 4]
        width = pred_reg[i, 3]

      xx1 = center_x - length / 2
      xx2 = center_x + length / 2
      yy1 = center_y - width / 2
      yy2 = center_y + width / 2

      for j in range(i + 1, num_candidate):
        if suppressed[j] == 1:
          continue
        com_center_x = pred_reg[j, 0]
        com_center_y = pred_reg[j, 1]
        rotate = pred_reg[j, 6]
        rotate_abs = c_abs(rotate % pi)
        if rotate_abs > (pi / 4) and rotate_abs < (pi * 3 / 4):
          com_length = pred_reg[j, 3]
          com_width = pred_reg[j, 4]
        else:
          com_length = pred_reg[j, 4]
          com_width = pred_reg[j, 3]

        com_xx1 = com_center_x - com_length / 2
        com_xx2 = com_center_x + com_length / 2
        com_yy1 = com_center_y - com_width / 2
        com_yy2 = com_center_y + com_width / 2

        left_x = max_float(xx1, com_xx1)
        right_x = min_float(xx2, com_xx2)
        top_y = max_float(yy1, com_yy1)
        bottom_y = min_float(yy2, com_yy2)

        intersection = max_float(0, right_x - left_x) * max_float(0, bottom_y - top_y)
        union = length * width + com_length * com_width

        if intersection / (union - intersection + 1e-5) > threshold:
          suppressed[j] = 1
    return result_index


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
def nms_2d(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max_float(ix1, x1[j])
            yy1 = max_float(iy1, y1[j])
            xx2 = min_float(ix2, x2[j])
            yy2 = min_float(iy2, y2[j])
            w = max_float(0.0, xx2 - xx1 + 1)
            h = max_float(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep
