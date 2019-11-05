import os
import shutil


# in_folder = '/Users/dekura/Desktop/opc/datasets/design_contour_paired/B'
# in_folder = '/Users/dekura/Desktop/opc/datasets/design_contour_paired_test/A'
in_folder = '/Users/dekura/Desktop/opc/datasets/design_sraf_contour_sraf_paired_rgb/B'


img_lists = os.listdir(in_folder)

img_lists = [img_path for img_path in img_lists if img_path.endswith('.png')]

total_img = len(img_lists)
print('total images nums: ', total_img)

for img_path in img_lists:
    img_rank = img_path.replace('via', '').replace('_mb_mb_lccout.oas.gds.png','')
    img_rank = int(img_rank)
    if img_rank <= 3572:
        old_path = os.path.join(in_folder, img_path)
        new_path = os.path.join(in_folder, 'test', img_path)
    else:
        old_path = os.path.join(in_folder, img_path)
        new_path = os.path.join(in_folder, 'train', img_path)
    shutil.move(old_path, new_path)
