#!/bin/bash
#SBATCH --job-name=met
#SBATCH --mail-user=cgjhaha@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d4/gds/gjchen21/tmp/log/dlsdmo_haoyu_metal_m2c_single_1024.txt
#SBATCH --gres=gpu:4
# this script is design to train a 2048*2048
# design come from the ganopc dataset
# the mask is generated from the deep levelset generator
# this use very large dataset

/research/d4/gds/gjchen21/miniconda3/envs/torch1.2/bin/python train.py \
--gpu_ids 0,1,2,3 \
--checkpoints_dir /research/d4/gds/gjchen21/github/dls-dmo/checkpoints \
--dataroot /research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/haoyu_data/metal_dls/ \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--niter 50 \
--niter_decay 50 \
--print_freq 100 \
--save_epoch_freq 10 \
--input_nc 1 \
--output_nc 1 \
--init_type kaiming \
--norm instance \
--name dlsdmo_haoyu_metal_m2c_single_1024 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0 \
--upp_scale 2