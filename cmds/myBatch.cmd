#!/bin/bash
#SBATCH --job-name=ganopc
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/ganopc_upp_base_50epoch.txt
#SBATCH --gres=gpu:2

/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1 \
--netG ganopc_unet \
--netD n_layers \
--n_layers_D 6 \
--pool_size 0 \
--batch_size 16 \
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