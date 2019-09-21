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
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from metrics import SegmentationMetric
from tqdm import tqdm

def tensor2grey(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            # image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy[image_numpy < 0] = 0
            image_numpy[image_numpy > 0] = 1
            image_numpy = image_numpy * 255.0
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_result2txt(path,pixAcc,mIoU):
    f = open(path,'a')
    f.write('testing time : {} \n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    f.write('pixAcc: {}, mIoU: {} \n \n'.format(pixAcc,mIoU))
    f.close()

def save_np2txt(arr,path):
    arr = arr[0][0].cpu().numpy()

    np.savetxt(path,arr)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # create a txt to save the testing results
    txt_path = web_dir + '/{}_epoch_{}.txt'.format(opt.name,opt.epoch)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    metric = SegmentationMetric(2)
    tbar = tqdm(dataset)
    if opt.eval:
        model.eval()
    finalAcc , finalmIoU = 0, 0
    for i, data in enumerate(tbar):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        if i >= 2:
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        """
        because the data in picture was divided into binary with [-1, 1]
        we need to set the label to [0,1]
        """
        gnd = data['B']
        pred =  visuals['fake_B']
        # print(img_path)
        (filepath,tempfilename) = os.path.split(img_path[0])
        (filename,extension) = os.path.splitext(tempfilename)

        base_path = '/research/dept7/glchen/tmp/debug/'
        gnd_path = base_path + filename + '_real_b.txt'
        pred_path = base_path + filename + '_fake_b.txt'
        # print(gnd_path)
        # print(pred_path)
        # save_np2txt(gnd, gnd_path)
        # save_np2txt(pred, pred_path)

        gnd[gnd < 0] = 0
        pred[pred < 0] = 0
        gnd[gnd > 0] = 1
        pred[pred > 0] = 1
        gnd = gnd.cpu().numpy()
        pred = pred.cpu().numpy()

        metric.update(pred, gnd)
        acc_cls, mean_iu = metric.get()
        tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (acc_cls, mean_iu))
        if i == opt.num_test - 1:
            finalAcc = acc_cls
            finalmIoU = mean_iu
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
    save_result2txt(txt_path, finalAcc, finalmIoU)
