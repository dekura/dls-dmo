###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2021-09-08 10:48:39
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 256 size image
 ###
/research/d4/gds/gjchen21/miniconda3/envs/torch1.2/bin/python test_mask_green.py \
--gpu_ids 0 \
--checkpoints_dir /research/d4/gds/gjchen21/github/dls-dmo/checkpoints \
--dataroot /research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/maskg_contourw_rect_paired_rgb_2048/haoyu_mc2 \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--results_dir /research/d4/gds/gjchen21/github/dls-dmo/results \
--name dlsdmo_haoyu_m2c_rgb_1024 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--eval \
--num_test 2170 \
--norm instance
