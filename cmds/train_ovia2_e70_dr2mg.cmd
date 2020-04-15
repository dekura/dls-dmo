#!/bin/bash
#SBATCH --job-name=dmo4
#SBATCH --mail-user=cgjhaha@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/ovia2_e70_dr2mg.txt
#SBATCH --gres=gpu:4

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3 \
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \
--dataroot /research/dept7/glchen/datasets/dlsopc_datasets/viasep/via2/dmo \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--niter 35 \
--niter_decay 35 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--name ovia2_e70_dr2mg \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0
