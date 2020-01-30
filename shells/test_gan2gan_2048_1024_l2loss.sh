###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2019-11-24 12:14:27
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 1024 size image
###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_l2.py \
--gpu_ids 0 \
--preprocess resize_and_crop \
--load_size 2048 \
--crop_size 2048 \
--input_nc 3 \
--output_nc 3 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--dataroot /research/dept7/glchen/datasets/gan_gds2png/gan2gan_100epoch_2048_1024_gl_sl1_fixed_100epoch \
--results_dir /research/dept7/glchen/github/pixel2pixel/results \
--name gan2gan_100epoch_2048_1024_gl_sl1_fixed_100epoch_l2loss \
--direction AtoB \
--dataset_mode unaligned \
--num_test 2170 \
--norm batch
