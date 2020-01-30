<!--
 * @Author: Guojin Chen
 * @Date: 2019-11-05 12:58:51
 * @LastEditTime: 2019-11-16 11:20:42
 * @Contact: cgjhaha@qq.com
 * @Description: readme of training
 -->
# Tips before you train


1. First clone this repo:

`https://github.com/dekura/pixel2pixel/`

2. read train_dls.md


datasets paths `/research/dept7/glchen/datasets/dlsopc_datasets` all is shared
checkpoints paths `/research/dept7/glchen/github/pixel2pixel/checkpoints` all is shared


# model path:

`/research/dept7/glchen/github/pixel2pixel/checkpoints/newdcupp_naive6_100epoch_mg2cw_2048_1024/latest_net_G.pth`
`/research/dept7/glchen/github/pixel2pixel/checkpoints/newdcupp_naive6_100epoch_mg2cw_2048_1024/latest_net_D.pth`

we need to firstly train an opc-gan, an the shell is `train_dcupp_naive6_dr2mg_2048_2048.sh`.
`dr2mg` this word means design-red-to-mask-green. `_2048_2048` the first 2048 means the images
in the datasets are 2048*2048size ,the second 2048 means we need to crop it into 2048 size.


# First: Train LITHOGAN

In this step we need to train a litho simulator which generate contour by the input mask.

you can run `bash train_newdcupp_naive6_mg2cw_2048_1024.sh`

mg2cw means: mask(green) to contour(white)

## shell
some important varibales:


```shell
/research/dept7/wlchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0,1,2,3 \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--dataroot /research/dept7/glchen/datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--netG new_dcupp \
--netD naive6_nl \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--niter 50 \
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--name newdcupp_naive6_100epoch_mg2cw_2048_1024 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0
```



## TEST Lithogan

shell

you can run `bash test_mg2cw_2048_1024_newdcupp.sh`
```shell
/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--load_size 1024 \
--crop_size 1024 \
--dataroot /research/dept7/glchen/datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB \
--checkpoints_dir /research/dept7/glchen/github/pixel2pixel/checkpoints \
--results_dir /research/dept7/glchen/github/pixel2pixel/results \
--name newdcupp_naive6_100epoch_mg2cw_2048_1024 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--dataset_mode aligned \
--eval \
--num_test 2170 \
--norm batch

```