#!/bin/bash
#SBATCH --job-name=20488
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/dcupp_naive6_100epoch_dr2mg_2048_1024.txt
#SBATCH --gres=gpu:8

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3,4,5,6,7 \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 8 \
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
--name dcupp_naive6_100epoch_dr2mg_2048_1024 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 400.0

