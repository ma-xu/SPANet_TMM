from __future__ import print_function

import torch
import argparse
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")




parser = argparse.ArgumentParser(description='Calculate the contributions')
parser.add_argument('--checkpoint', default='/home/g1007540910/checkpoints/imagenet/spa_resnet50/model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()


check_point = torch.load(args.checkpoint,map_location=torch.device('cpu'))

from collections import OrderedDict
new_check_point = OrderedDict()
for k, v in check_point['state_dict'].items():
    # name = k[7:]  # remove `module.`
    # name = k[9:]  # remove `module.1.`
    if k.startswith('module.1.'):
        name = k[9:]
    else:
        name = k[7:]
    if 'spa.weight' in name:
        print(v.view(3).data)
    new_check_point[name] = v


