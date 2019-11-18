###
# @Author: Guojin Chen
 # @Date: 2019-11-15 00:01:31
 # @LastEditTime: 2019-11-17 00:14:50
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 2 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--niter 25 \
--niter_decay 25 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_weighted_50epoch_dr2mg_2048_1024 \
--model pix2pixw \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0 \
--lambda_uppscale 4