#!/bin/bash
#SBATCH --job-name=p100
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/Custom_pix2pix_binary_100epoch_8batch_256_ouput.txt
#SBATCH --gres=gpu:1

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG custom \
--netD n_layers \
--pool_size 0 \
--batch_size 12 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 15 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name Custom_pix2pix_binary_100epoch_8batch_256 \
--model pix2pix \
--direction AtoB \
--display_id 0