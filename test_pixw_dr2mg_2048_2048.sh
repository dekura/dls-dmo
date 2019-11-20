### 
# @Author: Guojin Chen
 # @Date: 2019-11-18 00:28:12
 # @LastEditTime: 2019-11-20 16:56:33
 # @Contact: cgjhaha@qq.com
 # @Description: 
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD naive6_nl \
--display_winsize 2048 \
--preprocess resize_and_crop \
--load_size 2048 \
--crop_size 2048 \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_weighted_100epoch_dr2mg_2048_2048_uppscale4 \
--model pix2pixw \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch \
--lambda_uppscale 4