/research/dept7/glchen/miniconda3/envs/guojin/bin/python test.py \
--gpu_ids 0 \
--display_winsize 256 \
--preprocess resize_and_crop \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name dcr2att_unet_pix2pix_50epoch_bn \
--model pix2pix \
--netG dcr2att_unet \
--output_nc 1 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--epoch 50 \
--num_test 2170 \
--norm batch

