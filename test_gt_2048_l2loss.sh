###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2019-11-18 00:24:02
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 1024 size image
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python test_l2.py \
--gpu_ids 0 \
--preprocess resize_and_crop \
--batch_size 8 \
--load_size 2048 \
--crop_size 2048 \
--input_nc 3 \
--output_nc 3 \
--dataroot /research/dept7/glchen/datasets/design_contourw_paired_2048/combine_AB \
--results_dir /research/dept7/glchen/github/pixel2pixel/results \
--name calibre_2048_l2loss \
--direction AtoB \
--dataset_mode aligned \
--num_test 2170 \
--norm batch
