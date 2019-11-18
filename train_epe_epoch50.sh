### 
# @Author: Guojin Chen
 # @Date: 2019-11-16 22:00:10
 # @LastEditTime: 2019-11-18 14:50:48
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
--batch_size 8 \
--preprocess resize_and_crop \
--dataset_mode alignedabc \
--load_size 256 \
--crop_size 256 \
--niter 25 \
--niter_decay 25 \
--print_freq 500 \
--save_epoch_freq 25 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_mask_contour_ABC_2048/combine_AB \
--netG0_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_100epoch_dr2mg_2048/100_net_G.pth \
--netG_pretrained_path /research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_100epoch_mg2cw_2048/100_net_G.pth \
--name epe_50epoch_2048_256 \
--model epe \
--direction AtoB \
--display_id 0 \
--upp_scale 2