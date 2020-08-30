from __future__ import print_function

import torch
import argparse
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")




parser = argparse.ArgumentParser(description='Calculate the contributions')
parser.add_argument('--checkpoint', default='/Users/melody/Downloads/epoch_24.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--layer', default='spa.weight',type=str)
args = parser.parse_args()


check_point = torch.load(args.checkpoint,map_location=torch.device('cpu'))

from collections import OrderedDict
new_check_point = OrderedDict()

for k, v in check_point['state_dict'].items():
    # name = k[7:]  # remove `module.`
    # name = k[9:]  # remove `module.1.`

    if args.layer in k:
        print(v.view(3))
    new_check_point[k] = v




