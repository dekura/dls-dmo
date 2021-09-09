###
# @Author: Guojin Chen
 # @Date: 2019-11-15 00:01:31
 # @LastEditTime: 2021-09-08 09:34:56
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/d4/gds/gjchen21/miniconda3/envs/torch1.2/bin/python train.py \
--gpu_ids 0,1,2,3 \
--checkpoints_dir /research/d4/gds/gjchen21/github/dls-dmo/checkpoints \
--dataroot /research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 4 \
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
--norm instance \
--name dlsdmo_haoyu_m2c_rgb_1024 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0