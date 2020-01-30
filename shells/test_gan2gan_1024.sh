### 
# @Author: Guojin Chen
 # @Date: 2019-11-18 00:28:12
 # @LastEditTime: 2019-11-22 03:42:06
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
--dataroot /research/dept7/glchen/datasets/design_mask_contour_ABC_2048/combine_AB \
--name gan2gan_100epoch_2048_1024_gl_sl1_fixed \
--model gan2gan \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--epoch 50 \
--eval \
--num_test 2170 \
--norm batch \
--lambda_uppscale 4