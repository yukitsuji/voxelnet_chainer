#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import numpy as np
from chainer.training import extension


class PolynomialShift(extension.Extension):
    """Polynomial Shit """
    def __init__(self, power=0.9, stop_trigger=None, batchsize=4,
                 len_dataset=1, attr='lr'):
        self._attr = attr
        self._power = power
        self._init = None
        self._t = 0
        self._last_value = 0
        if stop_trigger.unit == 'iteration':
            self._maxiter = stop_trigger.period
        elif stop_trigger.unit == 'epoch':
            n_iter_per_epoch = len_dataset / float(batchsize)
            self._maxiter = float(stop_trigger.period * n_iter_per_epoch)

    def initialize(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

    def __call__(self, trainer):
        self._t += 1

        optimizer = trainer.updater.get_optimizer('main')
        value = self._init * ((1 - (self._t / self._maxiter)) ** self._power)
        setattr(optimizer, self._attr, value)
        self._last_value = value

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)
