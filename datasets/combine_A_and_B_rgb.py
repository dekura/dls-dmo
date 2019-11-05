import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--gray_B', dest='gray_B', help='input directory for image gray B', type=str, default='/Users/dekura/Desktop/opc/datasets/design_contour_paired/B')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='/Users/dekura/Desktop/opc/datasets/mask_sraf_contour_sraf_paired_rgb/A')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='/Users/dekura/Desktop/opc/datasets/mask_sraf_contour_sraf_paired_rgb/B')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='/Users/dekura/Desktop/opc/datasets/mask_sraf_contour_sraf_paired_rgb/combine_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
args = parser.parse_args()


args.use_AB = False

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))




splits = os.listdir(args.gray_B)

splits.remove('.DS_Store')



for sp in splits:
    img_fold_B = os.path.join(args.fold_B, sp)
    img_fold_A = os.path.join(args.fold_A, sp)
    img_gray_B = os.path.join(args.gray_B, sp)
    img_list = os.listdir(img_gray_B)
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]
    img_list = [img_path for img_path in img_list if img_path.endswith('.png')]
    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_B = img_list[n]
        path_B = os.path.join(img_fold_B, name_B)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            # name_A = name_B.replace('mb_mb_lccout.oas', 'mbsraf')
            name_A = name_B
        path_A = os.path.join(img_fold_A, name_A)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_B
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
        else:
            print('path A {} not exist'.format(path_A))