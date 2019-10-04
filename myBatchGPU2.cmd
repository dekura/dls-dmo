#!/bin/bash
#SBATCH --job-name=1l8
#SBATCH --mail-user=glchen@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/dcupp_1lD_8ndf_50epoch_output.txt
#SBATCH --gres=gpu:2

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1 \
--netG dc_unet_nested \
--netD n_layers \
--n_layers_D 1 \
--ndf 8 \
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
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name dcupp_1lD_8ndf_50epoch \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--upp_scale 2