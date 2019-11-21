### 
# @Author: Guojin Chen
 # @Date: 2019-11-18 22:57:16
 # @LastEditTime: 2019-11-18 23:42:44
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG ganopc_unet \
--netD n_layers \
--n_layers_D 6 \
--pool_size 0 \
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 25 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--name ganopc_upp_base_50epoch \
--model pix2pix \
--direction AtoB \
--display_id 0