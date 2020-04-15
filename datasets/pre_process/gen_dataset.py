'''
@Author: Guojin Chen
@Date: 2020-03-14 20:06:06
@LastEditTime: 2020-03-17 18:04:10
@Contact: cgjhaha@qq.com
@Description: get train test dataset by via
'''
import os
import glob
from tqdm import tqdm
from get_polys import get_poly_vianum
'''
@description: separate the given gds from different via num
@param {type} args {
    gds_path{str} : input gds_folder path
}
@return: list_by_via{
    '1': [] list of via_num = 1,
    ...
}
'''
def get_detail_list(args):
    restr = os.path.join(args.gds_path, '*.gds')
    gds_list = glob.glob(restr)
    gds_list.sort()
    list_by_via = {}
    for item in tqdm(gds_list):
        via_num = get_poly_vianum(item, args)
        if via_num:
            via_num = str(via_num)
            if not via_num in list_by_via:
                list_by_via[via_num] = []
            list_by_via[via_num].append(item)
    return list_by_via


def gen_dataset(args):
    test_data = []
    train_data = []
    list_by_via = get_detail_list(args)
    for via_num, via_num_list in list_by_via.items():
        print('via num {}: has {} data'.format(via_num, len(via_num_list)))
        max_pervia = min(len(via_num_list), args.max_pervia)
        via_num_list = via_num_list[:max_pervia]
        print('now set via num {}: to be {} data'.format(via_num, max_pervia))
        test_num = len(via_num_list)//args.test_ratio
        via_test = via_num_list[:test_num]
        via_train = via_num_list[test_num:]
        test_data = test_data + via_test
        train_data = train_data + via_train
    return test_data, train_data


'''
@description: separate the given gds from different via num
@param {type} args {
    gds_path{str} : input gds_folder path
}
@return: set_byvia{
    '1': ([],[]) list of via_num = 1, (test_set, train_set)
    ...
}
'''

def gen_set_byvia(args):
    list_by_via = get_detail_list(args)
    set_byvia = {}
    for via_num, via_num_list in list_by_via.items():
        if not via_num in args.gen_via_lists:
            continue
        print('via num {}: has {} data'.format(via_num, len(via_num_list)))
        max_pervia = min(len(via_num_list), args.max_pervia)
        via_num_list = via_num_list[:max_pervia]
        print('now set via num {}: to be {} data'.format(via_num, max_pervia))
        test_num = len(via_num_list)//args.test_ratio
        via_test = via_num_list[:test_num]
        via_train = via_num_list[test_num:]
        set_byvia[via_num] = (via_test, via_train)
    return set_byvia