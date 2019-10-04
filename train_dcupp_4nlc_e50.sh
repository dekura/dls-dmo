/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG dcupp_4nlc \
--netD n_layers \
--pool_size 0 \
--batch_size 1 \
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
--name dcupp_4nlc_pix2pix_50epoch \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--upp_scale 2 \
--no-use_canny_loss \
--no-use_vdsr_loss