###
# @Author: Guojin Chen
 # @Date: 2019-11-15 00:01:31
 # @LastEditTime: 2019-11-18 17:47:08
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3,4,5,6,7 \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 6 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 2048 \
--crop_size 2048 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 5 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_100epoch_dr2mg_2048_2048 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 200.0 \
--lambda_uppscale 8