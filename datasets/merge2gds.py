'''
@Author: Guojin Chen
@Date: 2019-11-18 00:36:43
@LastEditTime: 2019-11-18 17:10:11
@Contact: cgjhaha@qq.com
@Description: merge two gds file using calibre
'''
import os
import argparse
import shutil
from tqdm import tqdm

calibre_path = '/home/hgeng/Calibre/aoj_cal_2018.2_33.24/bin/calibre'

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='', type=str, help='experiment name')
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

def merge2gds(gds1_path, gds2_path, gds_path, cell_name):
    dir = os.path.abspath(os.path.dirname(__file__))
    rule_dir = RULE_PATH
    rule_dir = os.path.join(dir, rule_dir)

    oas2gds_rule = merge2oas_rule(gds1_path, gds2_path, gds_path)
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


"""
@gds1_path we need the design and sraf
@gds2_path we need the mask layer
"""

def merge2oas_rule(gds1_path, gds2_path, gds_path):
    drc_code = """
LAYOUT PATH    '%s'
""" % (gds1_path)
    drc_code += """
LAYOUT PRIMARY '*'
"""
    drc_code += """
DRC RESULTS DATABASE '%s' GDSII
""" % (gds_path)
    drc_code += """
PRECISION 1000
LAYOUT SYSTEM GDSII
DRC MAXIMUM RESULTS ALL
DRC MAXIMUM VERTEX 4000

"""
    drc_code +="""
LAYOUT PATH    '%s'
""" % (gds2_path)
    drc_code +="""


LAYER MAP 0   datatype 0 1000 LAYER target     1000
LAYER MAP 4   datatype 0 1002 LAYER lay_sraf   1002
LAYER MAP 9   datatype 0 1001 LAYER lay_opc    1001


// #01: output results
OUT_target     {COPY target     } DRC CHECK MAP OUT_target    0     0
OUT_opc        {COPY lay_opc    } DRC CHECK MAP OUT_opc       1     0
OUT_sraf       {COPY lay_sraf   } DRC CHECK MAP OUT_sraf      4     0
"""
    return drc_code


if __name__ == '__main__':

    # oas_folder = '/home/glchen/epetest_1024/results/dcupp_naive6_100epoch_dr2mg_2048_1024'
    gds1_folder = '/home/glchen/design_mask2gds/design_mask2gds_2048_gt/gds'
    gds2_folder = '/home/glchen/epeloss_design_mask2gds/epeloss_design_mask2gds_2048_256/gds/epe_50epoch_2048_256'
    gds_folder = '/home/glchen/datasets/merged_gds'

    gds_folder = os.path.join(gds_folder, name)
    if not os.path.exists(gds_folder):
        os.mkdir(gds_folder)
    gds1_list = os.listdir(gds1_folder)

    gds1_list = ['via303_rgb.gds', 'via1432_rgb.gds']
    # gds2_path = '/home/glchen/epeloss_design_mask2gds/epeloss_design_mask2gds_2048_256/gds/epe_50epoch_2048_256/via46_rgb.gds'
    # gds_path = '/home/glchen/datasets/merged_gds/via46_rgb.gds'
    for gds1 in tqdm(gds1_list):
        if gds1.endswith('.gds'):
            gds1_path = os.path.join(gds1_folder, gds1)
            gds2_path = os.path.join(gds2_folder, gds1)
            gds_path = os.path.join(gds_folder, gds1)
            cell_name = gds1.replace('.gds','')
            merge2gds(gds1_path, gds2_path, gds_path, cell_name)


    # shutil.rmtree(RULE_PATH)
    # shutil.rmtree(LOG_PATH)
    # os.remove(RULE_PATH)
    # os.remove(LOG_PATH)