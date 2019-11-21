###
# @Author: Guojin Chen
 # @Date: 2019-11-15 00:01:31
 # @LastEditTime: 2019-11-21 18:01:59
 # @Contact: cgjhaha@qq.com
 # @Description:
###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3,4,5,6,7 \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 8 \
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
--name newdcupp_naive6_100epoch_dr2mg_2048_2048_uppscale4 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 500.0 \
--lambda_uppscale 4