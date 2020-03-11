'''
@Author: Guojin Chen
@Date: 2020-03-09 20:19:13
@LastEditTime: 2020-03-11 21:25:09
@Contact: cgjhaha@qq.com
@Description: the gds to png flow

gds to png:
1. gds to (design + sraf) : (mask + design + sraf) : (wafer), all is rgb three channels
2. copy images to dataset for training
3. DMO: (design + sraf)+(mask + design + sraf);
4. DLS: (mask + design + sraf) : (wafer)
'''
import os
import glob
import argparse
from tqdm import tqdm
from consts import LAYERS, DIRS
from get_polys import get_polys
from gen_im import gen_d_m_s, gen_d_s, gen_w

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='layout_0.4', help='experiment name')
parser.add_argument('--size', type=int, default=1024, help='image size')
parser.add_argument('--for_dmo', default=False, action='store_true', help='for dmo')
parser.add_argument('--for_dls', default=False, action='store_true', help='for dls')
parser.add_argument('--gds_path', type=str, required=True, help='the input gds path')
parser.add_argument('--out_folder', type=str, required=True, help='output data folder')
# out_folder/dmo/trainA
# out_folder/dls/trainA
args = parser.parse_args()


def save_im(imA, imB, dataset_type, sub_type, item_name, args)
    pathA = os.path.join(args.out_folder, sub_type, dataset_type+'A')
    pathB = os.path.join(args.out_folder, sub_type, dataset_type+'B')
    pathA = os.path.join(pathA, item_name+'.png')
    pathB = os.path.join(pathB, item_name+'.png')
    imA.save(pathA)
    imB.save(pathA)

def gen_data(dataset, dataset_type, args):
    print('generating {} set'.format(dataset_type))
    for item in tqdm(dataset):
        item_name = os.path.splitext(os.path.basename(item))[0]
        polys = get_polys(item, args)
        if not polys:
            print('polys not found')
            continue
        dms = gen_d_m_s(polys, args)
        if args.for_dmo:
            ds = gen_d_s(polys, args)
            save_im(ds, dms, dataset_type, 'dmo', item_name, args)
        if args.for_dls:
            wafer = gen_w(polys, args)
            save_im(dms, wafer, dataset_type, 'dls', item_name, args)


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def prepare_subdirs(args, sub_type):
    sub_path = os.path.join(args.out_folder, sub_type)
    makedir(sub_path)
    for subsubdir in DIRS:
        subsubpath = os.path.join(sub_path, subsubdir)
        makedir(subsubpath)

def prepare_dirs(args):
    makedir(args.out_folder)
    if args.for_dmo:
        prepare_subdirs(args, 'dmo')
    if args.for_dls:
        prepare_subdirs(args, 'dls')

def main():
    gds_list = glob.glob("{}/*.gds".format(args.gds_path)).sort()
    test_num = len(gds_list)//4
    test_set = gds_list[0: test_num]
    train_set = gds_list[test_num:]
    prepare_dirs(args)
    gen_data(test_set, 'test', args)
    gen_data(train_set, 'train', args)

if __name__ == '__main__':
    main()
