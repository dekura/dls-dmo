/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_rgb.py \
--gpu_ids 0 \
--display_winsize 256 \
--preprocess resize_and_crop \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/design_contour_paired_rgb/combine_AB \
--name dcupp_naive6_50epoch_c3 \
--model pix2pix \
--netG dc_unet_nested \
--netD naive6_nl \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch
