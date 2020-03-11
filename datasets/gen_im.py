'''
@Author: Guojin Chen
@Date: 2020-03-11 17:40:47
@LastEditTime: 2020-03-11 19:36:11
@Contact: cgjhaha@qq.com
@Description: generate images
'''

from PIL import Image, ImageDraw
from consts import LAYERS


def gen_im_by_layer(polys, layers, args):
    clipsize = args.size
    im = Image.new("RGB", (clipsize, clipsize))
    draw = ImageDraw.Draw(im)
    for layer in layers:
        polyset = polys[layer]
        for j in range(0, len(polyset)):
            tmp = tuple(map(tuple, polyset[j]))
            if layer == 'design':
                draw.polygon(tmp, fill=(255, 0, 0))
            if layer == 'mask':
                draw.polygon(tmp, fill=(0, 255, 0))
            if layer == 'sraf':
                draw.polygon(tmp, fill=(0, 0, 255))
            if layer == 'wafer':
                draw.polygon(tmp, fill=(255, 255, 255))
    return im


'''
@description: generate design mask and sraf
@param {type}
@return: None
'''
def gen_d_m_s(polys, args):
    layers = ['design', 'mask', 'sraf']
    return gen_im_by_layer(polys, layers, args)



'''
@description: generate design and sraf
@param {type}
    polys {
        'layer name': polygons
    }
@return: None
'''
def gen_d_s(polys, args):
    layers = ['design', 'sraf']
    return gen_im_by_layer(polys, layers, args)


'''
@description: generate wafer
@param {type}
@return: None
'''
def gen_w():
    layers = ['wafer']
    return gen_im_by_layer(polys, layers, args)