#!/bin/bash
#SBATCH --job-name=mg2cw256
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/wlchen/guojin/tmp/dcupp_naive6_100epoch_mg2cw_2048_256.txt
#SBATCH --gres=gpu:4

/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 16 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_100epoch_mg2cw_2048_256 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0 \
--lambda_uppscale 2