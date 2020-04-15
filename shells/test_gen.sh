
#!/bin/bash
#SBATCH --job-name=dmo1
#SBATCH --mail-user=cgjhaha@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept7/glchen/tmp/log/layout_0.4_e100_dr2mg_1024_1024.txt
#SBATCH --gres=gpu:1

/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \
--dataroot /research/dept7/glchen/datasets/dlsopc_datasets/layout_0.4/dmo \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 1 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--niter 50 \
--niter_decay 50 \
--print_freq 100 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--name layout_0.4_e100_dr2mg_1024_1024 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0
