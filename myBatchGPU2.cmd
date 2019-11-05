#!/bin/bash
#SBATCH --job-name=nosraf
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/dcupp2wr_naive6_50epoch_c3.txt
#SBATCH --gres=gpu:2

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1 \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 25 \
--niter_decay 25 \
--print_freq 500 \
--save_epoch_freq 25 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_contour_paired_rgb/combine_AB \
--netG_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_50epoch_c3/50_net_G.pth \
--name dcupp2wr_naive6_50epoch_c3 \
--model stage2wr \
--direction AtoB \
--display_id 0 \
--upp_scale 2 \
--lambda_tanh_scale 1 \
--lambda_L1 1000.0