# 注意要先修改 dcu++ 中的scale为1
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test.py \
--gpu_ids 0 \
--display_winsize 256 \
--preprocess resize_and_crop \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name dcupp_pix2pix_50epoch \
--model pix2pix \
--netG dc_unet_nested \
--output_nc 1 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--epoch 50 \
--num_test 2170 \
--norm batch

