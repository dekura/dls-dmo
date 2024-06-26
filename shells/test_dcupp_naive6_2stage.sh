/research/dept7/glchen/miniconda3/envs/guojin/bin/python test.py \
--gpu_ids 0 \
--display_winsize 256 \
--preprocess resize_and_crop \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/design_contour_paired/combine_AB \
--name dcupp_dcupp_naive6_50epoch \
--model stage2 \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--output_nc 1 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch
