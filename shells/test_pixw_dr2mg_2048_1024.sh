### 
# @Author: Guojin Chen
 # @Date: 2019-11-18 00:28:12
 # @LastEditTime: 2019-11-18 00:30:10
 # @Contact: cgjhaha@qq.com
 # @Description: 
 ###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--load_size 1024 \
--crop_size 1024 \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_weighted_100epoch_dr2mg_2048_1024 \
--model pix2pixw \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch \
--lambda_uppscale 4