'''
@Author: Guojin Chen
@Date: 2019-11-11 22:10:48
@LastEditTime: 2019-11-14 22:28:20
@Contact: cgjhaha@qq.com
@Description: split images into test and train
'''
import os
import sys
import shutil
from tqdm import tqdm


# in_folder = '/Users/dekura/Desktop/opc/datasets/design_contour_paired/B'
# in_folder = '/Users/dekura/Desktop/opc/datasets/design_contour_paired_test/A'
# in_folder = '/Users/dekura/Desktop/opc/datasets/design_sraf_contour_sraf_paired_rgb/B'
# in_folder = '/Users/dekura/Desktop/opc/datasets/design_maskg_paired_rgb/B'
# in_folder = '/Users/dekura/Desktop/opc/datasets/maskg_contourw_paired_rgb/B'
# in_folder = '/Users/dekura/Desktop/opc/datasets/design_maskg_rect_paired_rgd/B'
# in_folder = '/Users/dekura/Desktop/opc/datasets/design_maskg_only_paired_rgb/B'

if len(sys.argv) > 1:
    in_folder = sys.argv[1]
    if not os.path.exists(in_folder):
        raise NotADirectoryError
else:
    raise ValueError

in_folder_test = 'test'
in_folder_test = os.path.join(in_folder, in_folder_test)
if not os.path.exists(in_folder_test):
    os.mkdir(in_folder_test)

in_folder_train = 'train'
in_folder_train = os.path.join(in_folder, in_folder_train)
if not os.path.exists(in_folder_train):
    os.mkdir(in_folder_train)

img_lists = os.listdir(in_folder)

img_lists = [img_path for img_path in img_lists if img_path.endswith('.png')]

total_img = len(img_lists)
print('total images nums: ', total_img)

for img_path in tqdm(img_lists):
    img_rank = img_path.replace('via', '').replace('_mbsraf_mb_mb_lccout.oas.gds.png','')
    img_rank = int(img_rank)
    if img_rank <= 3572:
        old_path = os.path.join(in_folder, img_path)
        new_path = os.path.join(in_folder, 'test', img_path)
    else:
        old_path = os.path.join(in_folder, img_path)
        new_path = os.path.join(in_folder, 'train', img_path)
    shutil.move(old_path, new_path)
