import cv2
import numpy
import pylab
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='import id please')
    parser.add_argument(
        '--id',
        dest='id',
        help='img id',
        default='000000001146'
    )
    parser.add_argument(
        '--nob',
        dest='nob',
        help='no boundary?',
        default=False
    )
    parser.add_argument(
        '--rootDir',
        dest='rootDir',
        help='root path of your image',
        default='/Users/dekura/Downloads/UNetNested_pix2pix_binary_100epoch_8batch_256/test_100/images/'
    )
    parser.add_argument(
        '--filePath',
        dest='filePath',
        help='using file path directly',
        default=False
    )
    return parser.parse_args()

# imgfile = input("请输入图片名：")
def saveLabel2Txt(id, nob, rootDir, filePath):
    if filePath:
        imgB = cv2.imread(filePath, cv2.IMREAD_COLOR)
        fnameB = open(filePath+'.txt','w')
        Xlenth = imgB.shape[1]#图片列数
        Ylenth = imgB.shape[0]#图片行数
        for i in range(Ylenth):
        # fname.write(str(a) + ':'+'\n')#----5
            for j in range(Xlenth):
                fnameB.write(str(imgB[i][j][0])+' ')
                # a += 1#----6
            fnameB.write('\n')
        fnameB.close()
        return
    imgfileB = 'via{}_mb_mb_lccout.oas.gds_fake_B.png'.format(id)
    imgfileA = 'via{}_mb_mb_lccout.oas.gds_real_A.png'.format(id)
    # txtfile = input("请输入存储文本文件名：")
    txtfileB = '{}_B.txt'.format(id)
    txtfileA = '{}_A.txt'.format(id)

    data_root = rootDir
    if nob:
        data_root = '/Users/dekura/Desktop/opc/Binary/train/train/'
    imgB = cv2.imread(data_root+imgfileB,cv2.IMREAD_COLOR)
    imgA = cv2.imread(data_root+imgfileA,cv2.IMREAD_COLOR)
    print("图像的形状,返回一个图像的(行数,列数,通道数):",imgB.shape)
    print("图像的像素数目:",imgB.size)
    print("图像的数据类型:",imgB.dtype)
    #----------------------------------------------------------------------------
    """
    In windows the COLOR->GRAYSCALE: Y = 0.299R+0.587G+0.114B 测试是否三个通道的值是相同的。
    某些图像三通道值相同，可以直接读灰度图来代替读单一通道。
    """
    # sum = 0
    # ans = 0
    # for i in range(562):
    #     for j in range(715):
    #         if not(img[i][j][0] == img[i][j][1] and img[i][j][1] == img[i][j][2]):
    #             sum += 1
    #         else:
    #             ans += 1
    # print(ans)
    # print(sum)
    #-----------------------------------------------------------------------------
    """
    将图片数据写入txt文件
    格式:
        基础信息
        行号:
            像素值
        行号:
            像素值
        ......
    """
    fnameB = open(data_root+txtfileB,'w')
    fnameA = open(data_root+txtfileA,'w')

    # fname.write("图像的形状,返回一个图像的(行数,列数,通道数):"+str(img.shape)+'\n')#----1
    # fname.write("图像的像素数目:"+str(img.size)+'\n')#----2
    # fname.write("图像的数据类型:"+str(img.dtype)+'\n')#----3
    Xlenth = imgB.shape[1]#图片列数
    Ylenth = imgB.shape[0]#图片行数
    # a = 1#----4
    for i in range(Ylenth):
        # fname.write(str(a) + ':'+'\n')#----5
        for j in range(Xlenth):
            fnameB.write(str(imgB[i][j][0])+' ')
        # a += 1#----6
        fnameB.write('\n')
    fnameB.close()

    for i in range(Ylenth):
        # fname.write(str(a) + ':'+'\n')#----5
        for j in range(Xlenth):
            fnameA.write(str(imgA[i][j][0])+' ')
        # a += 1#----6
        fnameA.write('\n')
    fnameA.close()
    #---------------------------------------------------------------------------
    """
    将txt文件中的数据读取进blist
    并显示为"test"图片框进行测试。
    注意进行测试前需要注释掉数据写入模块
    中标记的六行代码，要不然读取会出错误。
    """
    # blist = []
    # split_char = ' '
    # with open('C:/Users/Jake/Desktop/test01/'+txtfile, 'r') as bf:
    #     blist = [b.strip().split(split_char) for b in bf]
    #
    ##从txt文件读入进来的值类型是char需要转换为int
    # for i in range(Ylenth):
    #     for j in range(Xlenth):
    #         blist[i][j] = int(blist[i][j])
    #
    # tlist = numpy.array(blist)
    # plt.figure()
    # plt.imshow(tlist)
    # plt.axis('off') # 不显示坐标轴
    # pylab.show()
    #------------------------------------------------------------------------------
    """
    将图片显示在'image'图片框
    """
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #----------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    saveLabel2Txt(args.id, args.nob,args.rootDir,args.filePath)