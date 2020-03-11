###
# @Author: Guojin Chen
 # @Date: 2019-11-19 18:36:35
 # @LastEditTime: 2019-11-24 11:19:47
 # @Contact: cgjhaha@qq.com
 # @Description: data preparation for l2 loss test
 ###
python=/home/glchen/miniconda3/envs/py3/bin/python

# $python oas2gds.py --name ganopc_upp_base_50epoch --oas_folder /home/glchen/epetest_256/results/ganopc_upp_base_50epoch
# $python contourw2rgb.py --name ganopc_upp_base_50epoch --in_folder /home/glchen/datasets/gan_gds/ganopc_upp_base_50epoch
# $python design_sraf_2img.py --name ganopc_upp_base_50epoch --in_folder /home/glchen/datasets/gan_gds/ganopc_upp_base_50epoch

# $python oas2gds.py --name dcupp_naive6_100epoch_dr2mg_2048_512_gt --oas_folder /home/glchen/epetest_512/results/dcupp_naive6_100epoch_dr2mg_2048_512_gt/oas
# $python contourw2rgb.py --name dcupp_naive6_100epoch_dr2mg_2048_512_gt --in_folder /home/glchen/datasets/gan_gds/dcupp_naive6_100epoch_dr2mg_2048_512_gt
# $python design_sraf_2img.py --name dcupp_naive6_100epoch_dr2mg_2048_512_gt --in_folder /home/glchen/datasets/gan_gds/dcupp_naive6_100epoch_dr2mg_2048_512_gt


$python oas2gds.py --name ganopc_upp_base_25epoch --oas_folder /home/glchen/epetest_256/results/ganopc_upp_base_25epoch
$python contourw2rgb.py --name ganopc_upp_base_25epoch --in_folder /home/glchen/datasets/gan_gds/ganopc_upp_base_25epoch
$python design_sraf_2img.py --name ganopc_upp_base_25epoch --in_folder /home/glchen/datasets/gan_gds/ganopc_upp_base_25epoch