#!/bin/bash
#SBATCH --job-name=d2mn1
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/dcupp_naive_70epoch_c3_dr2mg_rect.txt
#SBATCH --gres=gpu:2

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1 \
--netG dc_unet_nested \
--netD naive_nl \
--pool_size 0 \
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 35 \
--niter_decay 35 \
--print_freq 500 \
--save_epoch_freq 35 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb/combine_AB \
--name dcupp_naive_70epoch_c3_dr2mg_rect \
--model pix2pix \
--direction AtoB \
--display_id 0