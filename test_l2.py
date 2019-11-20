'''
@Author: Guojin Chen
@Date: 2019-11-17 22:51:44
@LastEditTime: 2019-11-20 17:23:25
@Contact: cgjhaha@qq.com
@Description: test l2loss of datasets
'''
"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from tqdm import tqdm



def save_result2txt(path, l2loss, running_time):
    f = open(path,'a+')
    f.write('testing time : {} \n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    f.write('total running time: {}\n'.format(running_time))
    f.write('l2loss: {} \n'.format(l2loss))
    f.close()

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # create a website
    # create a txt to save the testing results
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    web_dir = os.path.join(opt.results_dir, opt.name)  # define the website directory
    if not os.path.exists(web_dir):
        os.mkdir(web_dir)
    txt_path = web_dir + '/{}_epoch_{}.txt'.format(opt.name,opt.epoch)
    tbar = tqdm(dataset)
    l2loss_mean = 0
    t = time.time()
    for i, data in enumerate(tbar):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        """
        because the data in picture was divided into binary with [-1, 1]
        we need to set the label to [0,1]
        """
        # print(data['A'].shape)
        if opt.input_nc == 1:
            design = data['A']
            wafer = data['B']
        else:
            red_layer = 0
            design = data['A'][:, red_layer]
            wafer = data['B'][:, red_layer]
        design = design+1
        design =  design/2
        wafer = wafer+1
        wafer = wafer/2
        before_sum = l2loss_mean * i
        now_sum = before_sum + torch.nn.MSELoss(reduction='sum')(wafer, design)
        l2loss_mean = now_sum/(i+1)

        tbar.set_description('l2loss: %.4f' % (l2loss_mean))
    elapsed = time.time() - t
    print('total running time: {}'.format(elapsed))
    save_result2txt(txt_path, l2loss_mean, elapsed)
