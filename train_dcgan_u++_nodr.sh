/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG dc_unet_nested \
--netD n_layers \
--pool_size 0 \
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 20 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name DCGANUNetNested_NODR_pix2pix_binary_100epoch_8batch_256 \
--model pix2pix \
--direction AtoB \
--display_id 0