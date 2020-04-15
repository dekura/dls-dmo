###
# @Author: Guojin Chen
 # @Date: 2019-11-16 11:22:32
 # @LastEditTime: 2020-03-13 11:14:27
 # @Contact: cgjhaha@qq.com
 # @Description: the shell to test 256 size image
 ###
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--load_size 1024 \
--crop_size 1024 \
--dataroot /research/dept7/glchen/datasets/dlsopc_datasets/via_layouts_0.4/dmo \
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \
--results_dir /research/dept7/glchen/github/dls-dmo/results \
--name via0.4_100epoch_dr2mg_1024_1024_g4 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--num_test 512 \
--eval \
--norm batch
