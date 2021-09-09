'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-09-08 09:49:43
LastEditTime: 2021-09-08 10:43:44
Contact: gjchen21@cse.cuhk.edu.hk
Description: generate the dataset for haoyu's requirements.
'''
import re
import shutil
from pathlib import Path

mc2_dir = '/research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/haoyu_mc2'
mc2_dir = Path(mc2_dir)

o_dir = '/research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/maskg_contourw_rect_paired_rgb_2048/combine_AB/'
o_dir = Path(o_dir)

o_test_dir = o_dir / 'test'

o_train_dir = o_dir / 'train'

new_test_dir = '/research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/maskg_contourw_rect_paired_rgb_2048/haoyu_mc2/test'
new_test_dir = Path(new_test_dir)

def gen_mc2():
    for p in mc2_dir.glob('*.png'):
        p_name = p.name
        print(p_name)
        file_id = re.findall(r"\d+", p_name)[0]
        o_test_name = f'via{file_id}_mb_mb_lccout.oas.gds.png'
        dst = new_test_dir / o_test_name
        src = o_test_dir / o_test_name
        if src.exists():
            shutil.copy(str(src), str(dst))
        else:
            src = o_train_dir /  o_test_name
            if src.exists():
                shutil.copy(str(src), str(dst))
            else:
                print(f'no such file {o_test_name}')



if __name__ == '__main__':
    gen_mc2()