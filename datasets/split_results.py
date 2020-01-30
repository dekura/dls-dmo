'''
@Author: Guojin Chen
@Date: 2019-11-14 12:29:14
@LastEditTime: 2019-11-14 13:45:18
@Contact: cgjhaha@qq.com
@Description: this file split the trained results to three png folder
    in order to get the gds file.
'''
import os
import shutil
from tqdm import tqdm



def split_results(in_folder, out_folder, test_convert_ids):
    if not os.path.exists(in_folder) or not os.path.exists(out_folder):
        print('in_floder: {}'.format(in_folder))
        print('out_folder: {}'.format(out_folder))
        raise NotADirectoryError
    if os.path.isfile(os.path.join(in_folder,'.DS_Store')):
        os.remove(os.path.join(in_folder,'.DS_Store'))
    # in_list = os.listdir(in_folder)
    for id in tqdm(test_convert_ids):
        img_real_A = 'via{}_mb_mb_lccout.oas.gds_real_A.png'.format(id)
        img_opc_A = 'via{}_mb_mb_lccout.oas.gds_fake_B.png'.format(id)
        img_real_B = 'via{}_mb_mb_lccout.oas.gds_real_B.png'.format(id)

        # copy file
        img_real_A_path = os.path.join(in_folder, img_real_A)
        img_opc_A_path = os.path.join(in_folder, img_opc_A)
        img_real_B_path = os.path.join(in_folder, img_real_B)

        shutil.copyfile(img_real_A_path, os.path.join(out_folder, 'real_A', img_real_A))
        shutil.copyfile(img_opc_A_path, os.path.join(out_folder, 'opc_A', img_opc_A))
        shutil.copyfile(img_real_B_path, os.path.join(out_folder, 'real_B', img_real_B))




in_folder = '/Users/dekura/Downloads/aresults/dcupp_naive6_100epoch_c3_dr2mg_only/test_latest/images/'
out_folder = '/Users/dekura/chen/py/design_mask2gds/png'


test_convert_ids = [
    1, 7, 11, 1003, 1008, 1009, 3421, 3002, 3004, 3006, 519, 975
]
if __name__ == '__main__':
    split_results(in_folder, out_folder, test_convert_ids)