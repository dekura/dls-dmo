 export http_proxy=http://proxy.cse.cuhk.edu.hk:8000/
 export TORCH_HOME=/research/dept7/glchen/.torch
/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1 \
--netG deeplabv3_plus \
--netD n_layers \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 25 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name deeplabv3_plus_pix2pix_100epoch \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--upp_scale 2 \
--no-use_canny_loss \
--no-use_vdsr_loss