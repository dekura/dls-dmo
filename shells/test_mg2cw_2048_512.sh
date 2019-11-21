###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2019-11-21 12:22:03
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 1024 size image
 ###
/research/dept7/glchen/phchen/miniconda3/envs/pytorch/bin/python test_mask_green.py \
--gpu_ids 0 \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 512 \
--preprocess resize_and_crop \
--load_size 512 \
--crop_size 512 \
--dataroot /research/dept7/glchen/datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--results_dir /research/dept7/glchen/github/pixel2pixel/results \
--name dcupp_naive6_100epoch_mg2cw_2048_512 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch
