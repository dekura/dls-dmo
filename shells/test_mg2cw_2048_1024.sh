###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2019-11-21 00:31:41
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 1024 size image
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--load_size 1024 \
--crop_size 1024 \
--dataroot /research/dept7/glchen/datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--results_dir /research/dept7/glchen/github/pixel2pixel/results \
--name gan2ganw_100epoch_2048_1024_ph_sl1 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--epoch 20 \
--num_test 2170 \
--norm batch \
--lambda_uppscale 4
