/research/dept7/wlchen/miniconda3/envs/pdenv/bin/python3 train.py \
--gpu_ids 0 \
--netG custom \
--netD n_layers \
--pool_size 0 \
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 40 \
--niter_decay 30 \
--print_freq 500 \
--save_epoch_freq 20 \
--output_nc 1 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/wlchen/dupeng/Binary \
--name Custom_pix2pix_binary_60epoch_256 \
--model pix2pix \
--direction AtoB \
--display_id 0
