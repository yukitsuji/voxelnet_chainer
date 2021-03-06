import numpy

import chainer

from chainer import function
from chainer import function_node
from chainer.utils import type_check

from chainer import configuration
from chainer import cuda
from chainer import functions
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn


class ActiveBatchNormalization(link.Link):

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(ActiveBatchNormalization, self).__init__()
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))
        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = batch_normalization(
                x, gamma, beta, eps=self.eps, running_mean=self.avg_mean,
                running_var=self.avg_var, decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0


class BatchNormalization(function_node.FunctionNode):

    mean = None
    inv_std = None

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9):
        self.running_mean = mean
        self.running_var = var

        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < 1e-5:
                msg = 'cuDNN does not allow an eps value less than 1e-5.'
                raise RuntimeError(msg)
        self.decay = decay

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, gamma, beta = inputs
        # x, gamma, beta, index1, index2 = inputs

        xp = cuda.get_array_module(x)
        if self.running_mean is None:
            self.running_mean = xp.zeros_like(gamma)
            self.running_var = xp.zeros_like(gamma)
        self.mode = _BNMode(x, gamma)

        # expander inserts singleton dimensions to gamma and beta so that they
        # can be broadcasted with x.
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        self.expander = expander
        self.axis = (0,) + tuple(range(head_ndim, x.ndim))

        gamma = gamma[expander]
        beta = beta[expander]
        # TODO: change by indexes
        self.mean = x.mean(axis=self.axis)
        var = x.var(axis=self.axis)

        var += self.eps
        self.inv_std = var ** (-0.5)
        y = _apply_bn_fwd(xp, x, self.mean[expander],
                          self.inv_std[expander], gamma, beta)
        # Update running statistics
        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        self.running_mean *= self.decay
        self.running_mean += (1 - self.decay) * self.mean
        self.running_var *= self.decay
        self.running_var += (1 - self.decay) * adjust * var

        return y,

    def backward(self, indexes, grad_outputs):
        x, gamma = self.get_retained_inputs()
        # x, gamma, index1, index2 = self.get_retained_inputs()
        gy, = grad_outputs

        # self.retain_inputs((0, 1, 2))
        # x, gamma, gy = inputs
        # x, gamma, index1, index2 = inputs

        expander = self.expander
        inv_m = gamma.dtype.type(1. / (x.size // gamma.size))
        xp = cuda.get_array_module(x)

        gbeta = gy.sum(axis=self.axis)
        x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])
        ggamma = (gy * x_hat).sum(axis=self.axis)
        if xp is numpy:
            gx = (gamma * self.inv_std)[expander] * (
                gy - (x_hat * ggamma[expander] + gbeta[expander]) * inv_m)
        else:
            gx = cuda.elementwise(
                '''
                T gy, T x_hat, T gamma, T inv_std, T ggamma, T gbeta,
                T inv_m
                ''',
                'T gx',
                '''
                gx = (gamma * inv_std) * (
                    gy - (x_hat * ggamma + gbeta) * inv_m)
                ''', 'bn_bwd')(gy, x_hat, gamma[expander],
                               self.inv_std[expander], ggamma[expander],
                               gbeta[expander], inv_m)
        self.retain_outputs((0, 1))
        return gx, ggamma, gbeta

        # f = BatchNormalizationGrad(
        #     self.eps, self.use_cudnn, self.mode, self.expander, self.axis,
        #     self.mean, self.inv_std)
        # return f(x, gamma, gy)


class BatchNormalizationGrad(function.Function):

    def __init__(self, eps, use_cudnn, mode, expander, axis, mean, inv_std):
        self.eps = eps
        self.use_cudnn = use_cudnn
        self.mode = mode
        self.expander = expander
        self.axis = axis
        self.mean = mean
        self.inv_std = inv_std

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, gamma, gy = inputs
        # x, gamma, index1, index2 = inputs

        expander = self.expander
        inv_m = gamma.dtype.type(1. / (x.size // gamma.size))
        xp = cuda.get_array_module(x)

        gbeta = gy.sum(axis=self.axis)
        x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])
        ggamma = (gy * x_hat).sum(axis=self.axis)
        if xp is numpy:
            gx = (gamma * self.inv_std)[expander] * (
                gy - (x_hat * ggamma[expander] + gbeta[expander]) * inv_m)
        else:
            gx = cuda.elementwise(
                '''
                T gy, T x_hat, T gamma, T inv_std, T ggamma, T gbeta,
                T inv_m
                ''',
                'T gx',
                '''
                gx = (gamma * inv_std) * (
                    gy - (x_hat * ggamma + gbeta) * inv_m)
                ''', 'bn_bwd')(gy, x_hat, gamma[expander],
                               self.inv_std[expander], ggamma[expander],
                               gbeta[expander], inv_m)
        self.retain_outputs((0, 1))
        return gx, ggamma, gbeta


class FixedBatchNormalization(function_node.FunctionNode):

    inv_std = None
    inv_var = None

    def __init__(self, eps=2e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 5)
        x_type, gamma_type, beta_type, mean_type, var_type = in_types
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
            mean_type.dtype == x_type.dtype,
            mean_type.shape == gamma_type.shape,
            var_type.dtype == x_type.dtype,
            var_type.shape == gamma_type.shape,
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1, 3, 4))
        x, gamma, beta, mean, var = inputs
        xp = cuda.get_array_module(x)

        # expander inserts singleton dimensions to gamma and beta so that they
        # can be broadcasted with x.
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        self.expander = expander
        self.axis = (0,) + tuple(range(head_ndim, x.ndim))

        mode = _BNMode(x, gamma)
        if mode.can_use_cudnn(xp):
            x = cuda.cupy.ascontiguousarray(x)

            gamma = cuda.cupy.ascontiguousarray(gamma)
            beta = cuda.cupy.ascontiguousarray(beta)
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            cudnn_mode = mode.get_cudnn_mode()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, cudnn_mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            y = cuda.cupy.empty_like(x)

            libcudnn.batchNormalizationForwardInference(
                handle, cudnn_mode, one.data, zero.data,
                x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
                derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
                mean.data.ptr, var.data.ptr, self.eps)
        else:
            gamma = gamma[expander]
            beta = beta[expander]
            var = var + self.eps
            self.inv_var = xp.reciprocal(var)
            self.inv_std = xp.sqrt(self.inv_var, dtype=self.inv_var.dtype)
            y = _apply_bn_fwd(xp, x, mean[expander], self.inv_std[expander],
                              gamma, beta)

        return y,

    def backward(self, indexes, grad_outputs):
        x, gamma, mean, var = self.get_retained_inputs()
        gy, = grad_outputs
        f = FixedBatchNormalizationGrad(
            self.eps, self.expander, self.axis, self.inv_std, self.inv_var)
        return f(x, gamma, mean, var, gy)


class FixedBatchNormalizationGrad(function.Function):

    def __init__(self, eps, expander, axis, inv_std, inv_var):
        self.eps = eps
        self.expander = expander
        self.axis = axis
        self.inv_std = inv_std  # may be None
        self.inv_var = inv_var  # may be None

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2, 4))
        x, gamma, mean, var, gy = inputs
        expander = self.expander
        xp = cuda.get_array_module(x)

        if self.inv_std is None or self.inv_var is None:
            self.inv_var = xp.reciprocal(var + self.eps)
            self.inv_std = xp.sqrt(self.inv_var, dtype=self.inv_var.dtype)

        self.gamma_over_std = gamma * self.inv_std
        x_hat = _x_hat(x, mean[expander], self.inv_std[expander])

        gx = self.gamma_over_std[expander] * gy
        gbeta = gy.sum(axis=self.axis)
        ggamma = (x_hat * gy).sum(axis=self.axis)
        gmean = -self.gamma_over_std * gbeta
        gvar = - 0.5 * gamma * self.inv_var * ggamma

        self.retain_outputs((0, 1, 2, 3, 4))
        return gx, ggamma, gbeta, gmean, gvar

    def backward(self, inputs, grad_outputs):
        x, gamma, mean, _, gy = inputs
        ggx1, gggamma1, ggbeta1, ggmean1, ggvar1 = grad_outputs
        gx1, ggamma1, gbeta1, gmean1, gvar1 = self.output_data

        # Handle None in output gradients.
        xp = cuda.get_array_module(x)
        ggx1 = _zero_if_none(xp, ggx1, x.shape, x.dtype)
        gggamma1 = _zero_if_none(xp, gggamma1, gamma.shape, gamma.dtype)
        ggbeta1 = _zero_if_none(xp, ggbeta1, gamma.shape, gamma.dtype)
        ggmean1 = _zero_if_none(xp, ggmean1, mean.shape, mean.dtype)
        ggvar1 = _zero_if_none(xp, ggvar1, mean.shape, mean.dtype)

        expander = self.expander

        x_hat = _x_hat(x, mean[expander], self.inv_std[expander])
        tmp = -0.5 * ggvar1

        gamma_over_var = gamma * self.inv_var
        g_gamma_over_var = tmp * ggamma1

        gggamma2 = gggamma1 + tmp * gamma_over_var
        gx_hat = gy * gggamma2[expander]
        gx2 = self.inv_std[expander] * gx_hat
        gmean2 = -self.inv_std * gx_hat.sum(axis=self.axis)

        g_gamma_over_std = (ggx1 * gy).sum(axis=self.axis) - ggmean1 * gbeta1
        ggbeta2 = ggbeta1 - ggmean1 * self.gamma_over_std
        ggy2 = (gggamma2[expander] * x_hat + ggbeta2[expander]
                + self.gamma_over_std[expander] * ggx1)

        ggamma2 = (self.inv_var * g_gamma_over_var
                   + self.inv_std * g_gamma_over_std)
        gvar2 = -(ggamma2 * gamma_over_var + 0.5 * self.inv_var * (
            (x_hat * gx_hat).sum(axis=self.axis)
            - self.gamma_over_std * g_gamma_over_std))

        return gx2, ggamma2, gmean2, gvar2, ggy2


class _BNMode(object):

    def __init__(self, x, gamma):
        is_gamma_1d = gamma.ndim == 1
        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        self.is_for_conv2d = x.ndim == 4 and is_gamma_1d
        self.is_for_linear = x.ndim == 2 and is_gamma_1d
        self.cudnn_dim_ok = self.is_for_conv2d or self.is_for_linear
        self.cudnn_dtype_ok = x.dtype != numpy.float16

    def get_cudnn_mode(self):
        assert self.cudnn_dim_ok
        if self.is_for_conv2d:
            return libcudnn.CUDNN_BATCHNORM_SPATIAL
        return libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

    def can_use_cudnn(self, xp):
        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        # cuDNN v5 batch normalization does not seem to support float16.
        return (xp is not numpy and
                chainer.should_use_cudnn('>=auto', 5000) and
                self.cudnn_dim_ok and
                self.cudnn_dtype_ok)


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


def _get_mode(x, gamma):
    if x.ndim == 4 and gamma.ndim == 1:
        return libcudnn.CUDNN_BATCHNORM_SPATIAL
    return libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION


def _x_hat(x, mean, inv_std):
    x_mu = x - mean
    x_mu *= inv_std
    return x_mu


def _apply_bn_fwd(xp, x, mean, inv_std, gamma, beta):
    # NOTE: all arguments should be broadcasted to x.shape
    # (mean, inv_std, gamma, and beta have to already be expanded)
    if xp is numpy:
        x_hat = _x_hat(x, mean, inv_std)
        y = gamma * x_hat
        y += beta
    else:
        y = cuda.elementwise(
            'T x, T mean, T inv_std, T gamma, T beta', 'T y',
            'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
        )(x, mean, inv_std, gamma, beta)
    return y


def _zero_if_none(xp, x, shape, dtype):
    # TODO(Tokui): Return broadcasted 0 instead of a zeroed array.
    if x is None:
        return xp.zeros(shape, dtype=dtype)
    return x


def batch_normalization(x, gamma, beta, **kwargs):
    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config')
    eps, running_mean, running_var, decay = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9))

    return BatchNormalization(eps, running_mean, running_var, decay).apply(
        (x, gamma, beta))[0]


def fixed_batch_normalization(x, gamma, beta, mean, var, eps=2e-5):
    return FixedBatchNormalization(eps).apply((x, gamma, beta, mean, var))[0]
