'''
@Author: Guojin Chen
@Date: 2020-03-16 11:29:23
@LastEditTime: 2020-03-19 17:34:03
@Contact: cgjhaha@qq.com
@Description: make command for train or testing
'''
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='layout_0.4', help='experiment name')
parser.add_argument('--file_name', type=str, default='dmo4.cmd', help='the cmd file name')
parser.add_argument('--gpu_num', type=int, default=4, help='how many gpu you need.')
parser.add_argument('--half_iter', type=int, default=35, help='half of total iter')
parser.add_argument('--p_freq', type=int, default=500, help='print frequence')
parser.add_argument('--dataroot', type=str, required=True, help='dataroot for train test')
parser.add_argument('--model', type=str, default='pix2pix', help='model pix2pix | pix2pixw')
parser.add_argument('--test_dmo', default=False, action='store_true', help='if gen test script')
parser.add_argument('--epoch', default='latest', type=str, help='test epoch')
parser.add_argument('--test_num', default=1000, type=int, help='test num')
parser.add_argument('--uppscale', default=2, type=int, help='uppscale lambda')
args = parser.parse_args()


def gen_dmo(args):
    gpu_ids = ','.join(str(x) for x in range(args.gpu_num))
    ckp_name = '{}_e{}_mg2cw'.format(args.name, 2*args.half_iter)
    dmo_code = """#!/bin/bash
#SBATCH --job-name=dls%d
#SBATCH --mail-user=cgjhaha@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/%s.txt
#SBATCH --gres=gpu:%d

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \\
--gpu_ids %s \\
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \\
--dataroot %s \\
--netG new_dcupp \\
--netD naive6_nl \\
--pool_size 0 \\
--batch_size %d \\
--preprocess resize_and_crop \\
--dataset_mode aligned \\
--load_size 1024 \\
--crop_size 1024 \\
--niter %d \\
--niter_decay %d \\
--print_freq %d \\
--save_epoch_freq 10 \\
--input_nc 3 \\
--output_nc 3 \\
--init_type kaiming \\
--norm batch \\
--name %s \\
--model %s \\
--direction AtoB \\
--display_id 0 \\
--lambda_L1 300.0 \\
--lambda_uppscale %d
"""%(args.gpu_num, ckp_name, args.gpu_num, gpu_ids, args.dataroot,
args.gpu_num, args.half_iter, args.half_iter, args.p_freq, ckp_name, args.model, args.uppscale)
    return dmo_code


def gen_test_dmo(args):
    gpu_ids = ','.join(str(x) for x in range(args.gpu_num))
    ckp_name = '{}_e{}_mg2cw'.format(args.name, 2*args.half_iter)
    test_code="""/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_mask_green.py \\
--gpu_ids %s \\
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \\
--dataroot %s \\
--netG new_dcupp \\
--netD naive6_nl \\
--display_winsize 1024 \\
--preprocess resize_and_crop \\
--dataset_mode aligned \\
--load_size 1024 \\
--crop_size 1024 \\
--results_dir /research/dept7/glchen/github/dls-dmo/results \\
--name %s \\
--model %s \\
--input_nc 3 \\
--output_nc 3 \\
--direction AtoB \\
--epoch %s \\
--num_test %d \\
--eval \\
--norm batch \\
--lambda_uppscale %d
"""%(gpu_ids, args.dataroot, ckp_name, args.model, args.epoch, args.test_num, args.uppscale)
    return test_code

def main(args):
    if args.test_dmo:
        code = gen_test_dmo(args)
    else:
        code = gen_dmo(args)
    print(code)
    with open(args.file_name, 'w+') as f:
        f.write(code)


if __name__ == '__main__':
    main(args)
