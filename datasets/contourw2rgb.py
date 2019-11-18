'''
@Author: Guojin Chen
@Date: 2019-11-10 18:37:33
@LastEditTime: 2019-11-18 12:14:34
@Contact: cgjhaha@qq.com
@Description: generate white contour in 2048*2048 size
'''
#gds2img.py by Haoyu
###########################
import gdspy
import sys
import os
import argparse
from PIL import Image, ImageDraw
from progress.bar import Bar
clipsize = 2048

DESIGN_LAYER = 0
OPC_LAYER = 1
SRAF_LAYER = 2
CONTOUR_LAYER = 200


def gds2img(Infolder, Infile, ImgOut):
    GdsIn = os.path.join(Infolder, Infile)
    gdsii   = gdspy.GdsLibrary(unit=1e-9)
    gdsii.read_gds(GdsIn,units='convert')
    cell    = gdsii.top_level()[0]
    bbox    = cell.get_bounding_box()
    opt_space=40 #Leave space at border in case of edge correction

    width = int((bbox[1,0]-bbox[0,0]))
    height= int((bbox[1,1]-bbox[0,1]))
    w_offset = int(bbox[0,0] - (clipsize-width)/2)
    h_offset = int(bbox[0,1] - (clipsize-height)/2)


    sellayer = [CONTOUR_LAYER] #Layer Number
    dtype = 0  #Layout Data Type
    polygon  = []
    im = Image.new("RGB", (clipsize, clipsize))
    draw = ImageDraw.Draw(im)
    token = 1
    for i in range(len(sellayer)):
        try:
            polyset = cell.get_polygons(by_spec=True)[(sellayer[i],dtype)]
        except:
            token=0
            print("Layer not found, skipping...")
            break
        for poly in range(0, len(polyset)):
            for points in range(0, len(polyset[poly])):
                polyset[poly][points][0]=int(polyset[poly][points][0]-w_offset)
                polyset[poly][points][1]=int(polyset[poly][points][1]-h_offset)
        for j in range(0, len(polyset)):
            tmp = tuple(map(tuple, polyset[j]))
            if sellayer[i] == DESIGN_LAYER:
                draw.polygon(tmp, fill=(255, 0, 0))
            if sellayer[i] == OPC_LAYER:
                draw.polygon(tmp, fill=(255, 0, 0))
            if sellayer[i] == SRAF_LAYER:
                draw.polygon(tmp, fill=(0, 0, 255))
            if sellayer[i] == CONTOUR_LAYER:
                draw.polygon(tmp, fill=(255, 255, 255))
    if token == 1:
        filename = Infile+".png"
        outpath  = os.path.join(ImgOut,filename)
        im.save(outpath)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='', type=str, help='experiment name')
args = parser.parse_args()
name = args.name

if name == '':
    print('please input a name')
    raise TypeError

# Infolder = sys.argv[1]
# Outfolder= sys.argv[2]

# Infolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_sample')
# Infolder = '/Users/dekura/Desktop/opc/design-april'
# Infolder = '/Users/dekura/Desktop/opc/datasets/lccout/gds'
# Infolder = '/Users/dekura/Desktop/opc/datasets/myresults/gds/'
Infolder = '/home/glchen/datasets/gan_gds/dcupp_naive6_100epoch_dr2mg_2048_1024'
# Outfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_sample_contour_output')
Outfolder = '/home/glchen/datasets/gan_gds2png'
Outfolder = os.path.join(Outfolder, name)
if not os.path.isdir(Outfolder):
    os.mkdir(Outfolder)

Outfolder = os.path.join(Outfolder, 'testB')
if not os.path.isdir(Outfolder):
    os.mkdir(Outfolder)

for dirname, dirnames, filenames in os.walk(Infolder):
    bar=Bar("Converting GDSII to Image", max=len(filenames))
    for f in range(0, len(filenames)):
        try:
            gds2img(Infolder, filenames[f], Outfolder)
        except:
            bar.next()
            continue
        bar.next()
bar.finish()