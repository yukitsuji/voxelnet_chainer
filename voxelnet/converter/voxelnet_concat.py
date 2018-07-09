#/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import six

from chainer import cuda


def voxelnet_concat(batch, device=None, padding=None):
    """Concatenates a list of examples into array(s).
    """
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(device, _concat_arrays(
                [example[i] for example in batch], padding[i])))
        del batch
        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(device, _concat_arrays(
                [example[key] for example in batch], padding[key]))
        del batch
        return result

    else:
        return to_device(device, _concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    if not isinstance(arrays[0], numpy.ndarray) and\
       not isinstance(arrays[0], cuda.ndarray):
        arrays = numpy.asarray(arrays)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        return xp.concatenate(arrays)

def to_device(device, x):
    """Send an array to a given device.
    """
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)
