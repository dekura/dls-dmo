'''
@Author: Guojin Chen
@Date: 2020-03-09 20:19:13
@LastEditTime: 2020-03-11 11:07:36
@Contact: cgjhaha@qq.com
@Description: the gds to png flow

gds to png:
1. gds to (design + sraf) : (mask + design + sraf) : (wafer), all is rgb three channels
2. concat images to dataset (design + sraf)+(mask + design + sraf); (mask + design + sraf) : (wafer)
'''

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='layout_0.4', help='experiment name')
parser.add_argument('--gds_path', type=str, required=True, help='the input gds path')

LAYER_NUM = {
    'design': 0,
    'mask': 1,
    'sraf': 2,
    'wafer': 200
}

