#!/bin/bash
#SBATCH --job-name=p2pTrain
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/wlchen/guojin/tmp/Custom_pix2pix_binary_60epoch_256_ouput.txt
#SBATCH --gres=gpu:1

/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG custom \
--netD n_layers \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 30 \
--niter_decay 30 \
--print_freq 500 \
--save_epoch_freq 15 \
--continue_train \
--epoch 45 \
--epoch_count 46 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/wlchen/guojin/dataset-opc/Binary \
--name Custom_pix2pix_binary_60epoch_256 \
--model pix2pix \
--direction AtoB \
--display_id 0
