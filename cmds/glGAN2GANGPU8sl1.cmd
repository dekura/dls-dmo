#!/bin/bash
#SBATCH --job-name=gan2sl1
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/gan2gan_100epoch_2048_1024_gl_sl1_fixed.txt
#SBATCH --gres=gpu:8

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
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
--continue_train \
--epoch 50 \
--epoch_count 51 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_mask_contour_ABC_2048/combine_AB \
--netG_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_100epoch_mg2cw_2048_1024/70_net_G.pth \
--name gan2gan_100epoch_2048_1024_gl_sl1_fixed \
--model gan2gan \
--direction AtoB \
--display_id 0 \
--lambda_uppscale 4 \
--lambda_L1 200.0 \
--lambda_OPC_L1 500.0 \
--lambda_EPE_L1 50.0