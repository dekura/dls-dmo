###
# @Author: Guojin Chen
 # @Date: 2019-11-18 00:28:12
 # @LastEditTime: 2019-11-24 16:56:48
 # @Contact: cgjhaha@qq.com
 # @Description:
###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_gan2gan.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--dataset_mode alignedabc \
--load_size 1024 \
--crop_size 1024 \
--dataroot /research/dept7/glchen/datasets/dataset_for_papaer/combine_AB \
--name checkpoints_for_paper \
--model gan2gan \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--eval \
--epoch 40 \
--num_test 2170 \
--norm batch \
--lambda_uppscale 4