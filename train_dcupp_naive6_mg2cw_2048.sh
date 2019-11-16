### 
# @Author: Guojin Chen
 # @Date: 2019-11-15 00:22:38
 # @LastEditTime: 2019-11-15 00:25:07
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 25 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_100epoch_mg2cw_2048 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0