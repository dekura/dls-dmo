#!/bin/bash
#SBATCH --job-name=p2pTest
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/wlchen/guojin/tmp/Custom_pix2pix_binary_60epoch_256_test_ouput.txt
#SBATCH --gres=gpu:1

/research/dept7/glchen/miniconda3/envs/guojin/bin/python test.py \
--gpu_ids 0\
--display_winsize 256 \
--preprocess resize_and_crop \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name Custom_pix2pix_binary_60epoch_256 \
--model pix2pix \
--netG unet_256 \
--output_nc 1 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch

