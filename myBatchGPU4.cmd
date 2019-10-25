#!/bin/bash
#SBATCH --job-name=512
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/dcupp_naive6_50epoch_512_output.txt
#SBATCH --gres=gpu:4

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3 \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 16 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 512 \
--crop_size 512 \
--niter 25 \
--niter_decay 25 \
--print_freq 500 \
--save_epoch_freq 15 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name dcupp_naive6_50epoch_512 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--upp_scale 2
