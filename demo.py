#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training

subprocess.call(['sh', "setup.sh"])

from voxelnet.config_utils import *

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

from voxelnet.converter.voxelnet_concat import voxelnet_concat

def demo_voxelnet():
    """Demo VoxelNet."""
    config, args = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    if args.gpu != -1:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    else:
        args.gpu = None

    dataset_config = config['dataset']['test']['args']
    index = 0
    for batch in test_iter:
        batch = voxelnet_concat(batch, args.gpu)
        model.predict(*batch, dataset_config)
        index += 1

def main():
    demo_voxelnet()

if __name__ == '__main__':
    main()
