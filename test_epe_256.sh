### 
# @Author: Guojin Chen
 # @Date: 2019-11-18 00:28:12
 # @LastEditTime: 2019-11-18 14:52:49
 # @Contact: cgjhaha@qq.com
 # @Description: 
 ###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_rgb.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--display_winsize 256 \
--preprocess resize_and_crop \
--dataset_mode alignedabc \
--load_size 256 \
--crop_size 256 \
--dataroot /research/dept7/glchen/datasets/design_mask_contour_ABC_2048/combine_AB \
--name epe_50epoch_2048_256 \
--model epe \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--eval \
--num_test 2170 \
--norm batch