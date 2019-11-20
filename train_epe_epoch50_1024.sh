### 
# @Author: Guojin Chen
 # @Date: 2019-11-16 22:00:10
 # @LastEditTime: 2019-11-19 14:48:40
 # @Contact: cgjhaha@qq.com
 # @Description:
 ###
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 1 \
--preprocess resize_and_crop \
--dataset_mode alignedabc \
--load_size 1024 \
--crop_size 1024 \
--niter 25 \
--niter_decay 25 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_mask_contour_ABC_2048/combine_AB \
--netG0_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_weighted_100epoch_dr2mg_2048_1024/latest_net_G.pth \
--netG_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_100epoch_mg2cw_2048_1024/50_net_G.pth \
--name epe_50epoch_2048_1024 \
--model epe \
--direction AtoB \
--display_id 0 \
--lambda_uppscale 4