#!/bin/bash
#SBATCH --job-name=dcsmall
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/dcupp_small_pix2pix_binary_150epoch_4batch_256_continue_output.txt
#SBATCH --gres=gpu:1

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD n_layers \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 75 \
--niter_decay 75 \
--print_freq 500 \
--save_epoch_freq 20 \
--continue_train \
--epoch 140 \
--epoch_count 141 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name dcupp_small_pix2pix_binary_150epoch_4batch_256 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--upp_scale 2
