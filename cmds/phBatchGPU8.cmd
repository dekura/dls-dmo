#!/bin/bash
#SBATCH --job-name=epel1ph
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/phchen/tmp/.txt
#SBATCH --gres=gpu:8

/research/dept7/glchen/phchen/miniconda3/envs/pytorch/bin/python train.py \
--gpu_ids 0,1,2,3,4,5,6,7 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode alignedabc \
--load_size 1024 \
--crop_size 1024 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 5 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_mask_contour_ABC_2048/combine_AB \
--netG0_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_weighted_100epoch_dr2mg_2048_1024/latest_net_G.pth \
--netG_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_100epoch_mg2cw_2048_1024/latest_net_G.pth \
--name epel1_50epoch_2048_1024_ph \
--model epel1 \
--direction AtoB \
--display_id 0 \
--lambda_uppscale 4 \
--lambda_L1 300.0 \
--lambda_EPE_L1 250.0 \
--lambda_R 100.0 \
--lambda_G 100.0 \
--lambda_B 100.0