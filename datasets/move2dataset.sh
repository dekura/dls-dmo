### 
# @Author: Guojin Chen
# @Date: 2019-11-14 22:09:17
 # @LastEditTime: 2019-11-14 22:59:19
# @Contact: cgjhaha@qq.com
# @Description: copy file to dest
###

# source_folder=/Users/dekura/Desktop/opc/datasets/myresults/design_sraf_2048/
# source_folder=/Users/dekura/Desktop/opc/datasets/myresults/maskg_designr_srafb_png_2048/
# source_folder=/Users/dekura/Desktop/opc/datasets/myresults/maskg_designr_srafb_png_2048/
# source_folder=/Users/dekura/Desktop/opc/datasets/myresults/contourw_2048/
# source_folder=/Users/dekura/Desktop/opc/datasets/myresults/design_sraf_2048/
source_folder=/Users/dekura/Desktop/opc/datasets/myresults/contourw_2048/

# target_folder=/Users/dekura/Desktop/opc/datasets/design_maskg_rect_paired_rgb_2048/A
# target_folder=/Users/dekura/Desktop/opc/datasets/design_maskg_rect_paired_rgb_2048/B
# target_folder=/Users/dekura/Desktop/opc/datasets/maskg_contourw_rect_paired_rgb_2048/A
# target_folder=/Users/dekura/Desktop/opc/datasets/maskg_contourw_rect_paired_rgb_2048/B
target_folder=/Users/dekura/Desktop/opc/datasets/design_contourw_paired_2048/B

python=/usr/local/miniconda3/envs/pytorch/bin/python


cp -r $source_folder $target_folder

$python move2test.py $target_folder



