###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2019-11-22 03:15:42
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 256 size image
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--load_size 1024 \
--crop_size 1024 \
--results_dir /research/dept7/glchen/github/pixel2pixel/results \
--name newdcupp_naive6_100epoch_dr2mg_2048_1024 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch
