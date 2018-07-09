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

class SpatialDropout(ParentClass):

    """SpatialDropout regularization."""

    def __init__(self, dropout_ratio):
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError('dropout_ratio must be in the range [0, 1)')
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        if hasattr(self, 'mask'):
            y = x[0] * self.mask
        else:
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            xp = cuda.get_array_module(*x)
            mask = xp.ones(x[0].shape, dtype=numpy.float32)
            rand = xp.random.rand(*x[0].shape[:2])
            mask[rand <= self.dropout_ratio] = 0

            if xp == numpy:
                self.mask = mask * scale
                y = x[0] * self.mask
            else:
                self.mask, y = cuda.elementwise(
                    'T x, T mask1, T scale', 'T mask, T y',
                    '''
                    mask = mask1 * scale;
                    y = x * mask;
                    ''',
                    'spatial_dropout_fwd',
                )(x[0], mask, scale)
        return y,

    def backward(self, x, gy):
        return SpatialDropoutGrad(self.mask).apply(gy)


class SpatialDropoutGrad(ParentClass):
    """Computes the gradient of the SpatialDropout function."""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, inputs):
        y = inputs[0] * self.mask
        return y,

    def backward(self, indexes, gy):
        return SpatialDropoutGrad(self.mask).apply(gy)


def spatial_dropout(x, ratio=.1, **kwargs):
    """spatial_dropout(x, ratio=.1)"""
    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config')
    argument.assert_kwargs_empty(kwargs)

    if configuration.config.train:
        return SpatialDropout(ratio).apply((x,))[0]
    return chainer.as_variable(x)
