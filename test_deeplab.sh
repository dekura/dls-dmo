export TORCH_HOME=/research/dept7/glchen/.torch
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test.py \
--gpu_ids 0 \
--display_winsize 256 \
--preprocess resize_and_crop \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/dataset-opc/Binary \
--name deeplabv3_plus_pix2pix_100epoch \
--model pix2pix \
--netG deeplabv3_plus \
--output_nc 1 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--epoch 100 \
--num_test 2170 \
--norm batch

