<!--
 * @Author: Guojin Chen
 * @Date: 2019-11-05 12:58:51
 * @LastEditTime: 2019-11-16 11:20:42
 * @Contact: cgjhaha@qq.com
 * @Description: readme of training
 -->
# Tips before you train


we need to firstly train an opc-gan, an the shell is `train_dcupp_naive6_dr2mg_2048_2048.sh`.
`dr2mg` this word means design-red-to-mask-green. `_2048_2048` the first 2048 means the images
in the datasets are 2048*2048size ,the second 2048 means we need to crop it into 2048 size.

## shell

```shell
/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \                               you can change it to multigpu 0,1,2,3,4,5,6,7
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 1 \                            change this carefully
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 2048 \
--crop_size 2048 \
--niter 50 \                                total epoch = niter+niter_decay = 50+50=100
--niter_decay 50 \
--print_freq 500 \
--save_epoch_freq 10 \
--input_nc 3 \
--output_nc 3 \
--init_type kaiming \
--norm batch \
--dataroot /research/dept7/glchen/datasets/design_maskg_rect_paired_rgb_2048/combine_AB \
--name dcupp_naive6_100epoch_dr2mg_2048_2048 \
--model pix2pix \
--direction AtoB \
--display_id 0 \
--lambda_L1 300.0
```

# This is the original design.
# First: Train LITHOGAN

## shell

You can use `train_dcupp_naive6_c3.sh` in root directory to train.

some important varibales:

```shell
/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG dc_unet_nested \        can not change
--netD naive6_nl \             can not change
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 256 \
--crop_size 256 \
--niter 25 \                   control epoch
--niter_decay 25 \             control epoch
--print_freq 500 \
--save_epoch_freq 25 \
--input_nc 3 \                 use rgb, must be 3
--output_nc 3 \                use rgb, must be 3
--init_type kaiming \
--norm batch \
--dataroot /path/to/your/mask/dataroot \ modify before you run, must be mask dataset
--name dcupp_naive6_50epoch_c3 \ 
--model pix2pix \    when you train gan, you must use pix2pix model
--direction AtoB \
--display_id 0
```





# Second: Train FCN model to get opc mask

## shell

You can use `train_dcupp2wr_epoch50_naive6_c3_t1.sh` in root directory to train.

~~dcupp2wr:means dcupp + 2stage + weighted l1loss + recurrent~~



some important varibales:

```shell
/research/dept7/glchen/miniconda3/envs/guojin/bin/python train.py \
--gpu_ids 0 \
--netG0 dc_unet_nested \
--netG dc_unet_nested \
--netD naive6_nl \
--pool_size 0 \
--batch_size 4 \
--preprocess resize_and_crop \
--dataset_mode aligned \
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
--dataroot /path/to/design/dataset \ 				!!!different from mask dataset
--netG_pretrained_path where/is/your/gan/model \
--name dcupp2wr_naive6_50epoch_c3 \
--model stage2wr \                          must be stage2wr
--direction AtoB \
--display_id 0 \
--upp_scale 2 \
--lambda_tanh_scale 5 \                     the lambda tanh scale in FCN
--lambda_L1 100.0 \                         L1 weight 
--lambda_R 15                   the opc mask l1 loss weight for red layer
```

