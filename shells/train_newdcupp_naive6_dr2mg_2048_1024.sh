###
# @Author: Guojin Chen
 # @Date: 2019-11-15 00:01:31
 # @LastEditTime: 2019-11-21 02:42:13
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 1 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--name newdcupp_naive6_100epoch_dr2mg_2048_1024 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0