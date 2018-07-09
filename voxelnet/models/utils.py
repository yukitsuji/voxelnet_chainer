#/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer


def create_timer():
    start = chainer.cuda.Event()
    stop = chainer.cuda.Event()
    start.synchronize()
    start.record()
    return start, stop

def print_timer(start, stop, sentence="Time"):
    stop.record()
    stop.synchronize()
    elapsed_time = chainer.cuda.cupy.cuda.get_elapsed_time(
                           start, stop) / 1000
    print(sentence, elapsed_time)
    return elapsed_time
