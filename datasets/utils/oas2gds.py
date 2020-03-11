'''
@Author: Guojin Chen
@Date: 2019-11-18 00:36:43
@LastEditTime: 2019-11-19 18:37:52
@Contact: cgjhaha@qq.com
@Description: oas2gds using calibre
'''
import os
import argparse
import shutil
from tqdm import tqdm

calibre_path = '/home/hgeng/Calibre/aoj_cal_2018.2_33.24/bin/calibre'

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='', type=str, help='experiment name')
parser.add_argument('--oas_folder', default='', type=str, help='the oas folder path')
args = parser.parse_args()
name = args.name

if name == '':
    print('please input experiment name')
    raise EnvironmentError
dir = os.path.abspath(os.path.dirname(__file__))
RULE_PATH = './rule_'+name
RULE_PATH = os.path.join(dir, RULE_PATH)
LOG_PATH = './log_'+name
LOG_PATH = os.path.join(dir, LOG_PATH)
if not os.path.exists(RULE_PATH):
    os.mkdir(RULE_PATH)
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

def oas2gds(oas_path, gds_path, cell_name):
    dir = os.path.abspath(os.path.dirname(__file__))
    rule_dir = RULE_PATH
    rule_dir = os.path.join(dir, rule_dir)

    oas2gds_rule = make_oas2gds_rule(oas_path, gds_path)
    rule_name = cell_name + '_oas2gds_rule.cali'
    rule_path = os.path.join(rule_dir, rule_name)
    fs = open(rule_path, 'w+')
    fs.write(oas2gds_rule)
    fs.close()

    log_dir = os.path.join(dir, LOG_PATH)
    log_name = cell_name + '_oas2gds.log'
    log_path = os.path.join(log_dir, log_name)
    cmd = '{} -drc -hier -64 {} >& {}'.format(calibre_path, rule_path, log_path)
    # print cmd
    os.system(cmd)



def make_oas2gds_rule(oas_path, gds_path):
    drc_code = """
LAYOUT PATH    '%s'
""" % (oas_path)
    drc_code += """
LAYOUT PRIMARY '*'
"""
    drc_code += """
DRC RESULTS DATABASE '%s' GDSII
""" % (gds_path)
    drc_code += """
PRECISION 1000
LAYOUT SYSTEM OASIS
DRC MAXIMUM RESULTS ALL
DRC MAXIMUM VERTEX 4000

LAYER MAP 0   datatype 0 1000 LAYER target     1000
LAYER MAP 1   datatype 0 1001 LAYER lay_opc    1001
LAYER MAP 2   datatype 0 1002 LAYER lay_sraf   1002
LAYER MAP 200  datatype 0 1003 LAYER lay_contour  1003

// #01: output results
OUT_target     {COPY target     } DRC CHECK MAP OUT_target    0     0
OUT_opc        {COPY lay_opc    } DRC CHECK MAP OUT_opc       1     0
OUT_sraf       {COPY lay_sraf   } DRC CHECK MAP OUT_sraf      2     0
OUT_contour      {COPY lay_contour  } DRC CHECK MAP OUT_contour   200    0
"""
    return drc_code


if __name__ == '__main__':

    # oas_folder = '/home/glchen/epetest_1024/results/dcupp_naive6_100epoch_dr2mg_2048_1024'
    # oas_folder = '/home/glchen/epetest_weighted_1024/results/dcupp_naive6_weighted_100epoch_dr2mg_2048_1024_gt/oas'
    # oas_folder = '/home/glchen/epetest_256/results/ganopc_upp_base_50epoch'
    oas_folder = args.oas_folder
    gds_folder = '/home/glchen/datasets/gan_gds'
    gds_folder = os.path.join(gds_folder, name)
    if not os.path.exists(gds_folder):
        os.mkdir(gds_folder)
    oas_list = os.listdir(oas_folder)

    for oas in tqdm(oas_list):
        if oas.endswith('GAN.oas'):
            oas_path = os.path.join(oas_folder, oas)
            gds_path = os.path.join(gds_folder, oas+'.gds')
            cell_name = oas.replace('.oas','')
            oas2gds(oas_path, gds_path, cell_name)

    shutil.rmtree(RULE_PATH)
    shutil.rmtree(LOG_PATH)
    # os.remove(RULE_PATH)
    # os.remove(LOG_PATH)