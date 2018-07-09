import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check

import numpy

import chainer
from chainer import configuration
from chainer import cuda
try:
    from chainer import function_node
    ParentClass = function_node.FunctionNode
except:
    from chainer import function
    ParentClass = function.Function
from chainer.utils import argument
from chainer.utils import type_check


class FeatureToVoxel(ParentClass):

    """FeatureToVoxel"""

    def __init__(self, k, d, h, w):
        self.k, self.d, self.h, self.w = k, d, h, w
        self.batch_indexes = None
        self.d_indexes = None
        self.h_indexes = None
        self.w_indexes = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        feature, voxel_indexes, batch = inputs
        xp = cuda.get_array_module(feature)
        batch_k, channels = feature.shape
        voxel_indexes = xp.array(voxel_indexes, dtype="i")
        # batch = int(batch_k / self.k)
        # self.batch_indexes = xp.repeat(xp.arange(batch), self.k)
        self.batch_indexes = xp.empty(batch_k)
        batchsize = batch.shape[0]
        batch = xp.cumsum(batch)
        self.batch_indexes[0 : batch[0]] = 0
        for b in range(batchsize - 1):
            self.batch_indexes[batch[b] : batch[b+1]] = b + 1
        self.batch_indexes = self.batch_indexes.astype('i')
        self.d_indexes = (voxel_indexes / (self.h * self.w)).astype("i")
        self.h_indexes = ((voxel_indexes % (self.h * self.w)) / self.w).astype('i')
        self.w_indexes = ((voxel_indexes % (self.h * self.w)) % self.w).astype('i')
        voxel = xp.zeros((batchsize, channels, self.d, self.h, self.w),
                            dtype=numpy.float32)
        voxel[self.batch_indexes, :, self.d_indexes, self.h_indexes, self.w_indexes] =\
            feature # (B * K, 128)
        voxel[:, :, 0, 0, 0] = 0
        return voxel,

    def backward(self, x, gy):
        out = gy[0]
        # out.data[:, :, 0, 0, 0] = 0
        out = out[self.batch_indexes, :, self.d_indexes, self.h_indexes, self.w_indexes]
        return out,
        # return gy[0][self.batch_indexes, :, self.d_indexes, self.h_indexes, self.w_indexes],


def feature_to_voxel(x, voxel_indexes, k, d, h, w, batch):
    """Feature to Voxel function."""
    return FeatureToVoxel(k, d, h, w).apply((x, voxel_indexes, batch))[0]
