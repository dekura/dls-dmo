###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2021-09-08 19:13:36
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 256 size image
 ###
/research/d4/gds/gjchen21/miniconda3/envs/torch1.2/bin/python test.py \
--gpu_ids 0 \
--checkpoints_dir /research/d4/gds/gjchen21/github/dls-dmo/checkpoints \
--dataroot /research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/ispdhaoyusep_single/dls \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--results_dir /research/d4/gds/gjchen21/github/dls-dmo/results \
--name dlsdmo_haoyu_m2c_single_1024 \
--model pix2pix \
--input_nc 1 \
--output_nc 1 \
--direction AtoB \
--eval \
--num_test 200000 \
--norm instance
