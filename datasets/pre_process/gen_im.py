'''
@Author: Guojin Chen
@Date: 2020-03-11 17:40:47
@LastEditTime: 2020-03-15 14:57:53
@Contact: cgjhaha@qq.com
@Description: generate images
'''
from PIL import Image, ImageDraw
from consts import LAYERS


def gen_im_by_layer(polys, layers, args):
    clipsize = args.load_size
    im = Image.new("RGB", (clipsize, clipsize))
    im_r = im.getchannel('R')
    im_g = im.getchannel('G')
    im_b = im.getchannel('B')
    draw_r = ImageDraw.Draw(im_r)
    draw_g = ImageDraw.Draw(im_g)
    draw_b = ImageDraw.Draw(im_b)
    for layer in layers:
        polyset = polys[layer]
        for j in range(0, len(polyset)):
            tmp = tuple(map(tuple, polyset[j]))
            if layer == 'design':
                draw_r.polygon(tmp, fill=255)
            if layer == 'mask':
                draw_g.polygon(tmp, fill=255)
            if layer == 'sraf':
                draw_b.polygon(tmp, fill=255)
            if layer == 'wafer':
                draw_r.polygon(tmp, fill=255)
                draw_g.polygon(tmp, fill=255)
                draw_b.polygon(tmp, fill=255)
    im = Image.merge('RGB', [im_r, im_g, im_b])
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
def gen_w(polys, args):
    layers = ['wafer']
    return gen_im_by_layer(polys, layers, args)