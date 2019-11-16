#gds2img.py by Guojin~
# this make the design to be the center of the output image
###########################
import gdspy
import os
import numpy as np
from PIL import Image, ImageDraw
from progress.bar import Bar
clipsize = 2048

def gds2img(Infolder, Infile, ImgOut):
    GdsIn = os.path.join(Infolder, Infile)
    gdsii   = gdspy.GdsLibrary(unit=1e-9)
    gdsii.read_gds(GdsIn,units='convert')
    cell    = gdsii.top_level()[0]

    token = 1

    mask_bbox, token = get_bbox_by_mask(cell, token)
    if token == 0:
        print('mask_bbox_layer not found...')
        return
    # bbox    = cell.get_bounding_box()
    bbox = mask_bbox
    # opt_space=40 #Leave space at border in case of edge correction

    width = int((bbox[1, 0]-bbox[0, 0]))
    height = int((bbox[1, 1]-bbox[0, 1]))
    w_offset = int(bbox[0, 0] - (clipsize-width)/2)
    h_offset = int(bbox[0, 1] - (clipsize-height)/2)


    sellayer = [2, 20] #Layer Number
    dtype = 0  #Layout Data Type
    # polygon  = []
    im = Image.new('1', (clipsize, clipsize))
    draw = ImageDraw.Draw(im)

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
            draw.polygon(tmp, fill=255)

    if token == 1:
        filename = Infile+".png"
        outpath  = os.path.join(ImgOut,filename)
        im.save(outpath)


def get_bbox_by_mask(cell, token):
    mask_bbox_layer = 21
    dtype = 0
    try:
        polyset = cell.get_polygons(by_spec=True)[(mask_bbox_layer, dtype)]
    except:
        token = 0
    min_polygons = []
    max_polygons = []
    for poly in range(0, len(polyset)):
        if poly == 0:
            min_polygons = np.array([polyset[poly].min(0)])
            max_polygons = np.array([polyset[poly].max(0)])
        else:
            tmp_min = np.array([polyset[poly].min(0)])
            tmp_max = np.array([polyset[poly].max(0)])
            min_polygons = np.concatenate((min_polygons, tmp_min))
            max_polygons = np.concatenate((max_polygons, tmp_max))
    output = np.array([min_polygons.min(0), max_polygons.max(0)])
    return output, token

# Infolder = sys.argv[1]
# Outfolder= sys.argv[2]

# Infolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_sample')
Infolder = '/Users/dekura/Desktop/opc/datasets/design-april'
# Outfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_sample_output')
Outfolder = '/Users/dekura/Desktop/opc/datasets/design-april-png-center'

for dirname, dirnames, filenames in os.walk(Infolder):
    bar=Bar("Converting GDSII to Image", max=len(filenames))
    for f in range(0, len(filenames)):
        try:
            if filenames[f].endswith('gds'):
                gds2img(Infolder, filenames[f], Outfolder)
        except:
            bar.next()
            continue
        bar.next()
bar.finish()