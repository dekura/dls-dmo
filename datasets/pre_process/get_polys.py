'''
@Author: Guojin Chen
@Date: 2020-03-11 14:12:16
@LastEditTime: 2020-03-15 14:57:03
@Contact: cgjhaha@qq.com
@Description: get polys from a gds file.
'''
#gds2img.py by Haoyu
###########################
import gdspy
from consts import LAYERS


'''
@description: get polys of gds file
@param {type}
    infile str
    args argparse.args
@return: polys {
    'design': [],
    'sraf': [],
    'mask': [],
    'wafer': []
}
'''
def get_polys(infile, args):
    clipsize = args.load_size
    dtype = 0
    gdsii = gdspy.GdsLibrary(unit=1e-9)
    gdsii.read_gds(infile,units='convert')
    cell = gdsii.top_level()[0]
    bbox = cell.get_bounding_box()
    width = int((bbox[1,0]-bbox[0,0]))
    height= int((bbox[1,1]-bbox[0,1]))
    w_offset = int(bbox[0,0] - (clipsize-width)/2)
    h_offset = int(bbox[0,1] - (clipsize-height)/2)
    polys = {}
    for name, num in LAYERS.items():
        try:
            polyset = cell.get_polygons(by_spec=True)[(num,dtype)]
        except:
            print('layer {}:{} not found, skipping...'.format(name, num))
            return None
        for poly in range(0, len(polyset)):
            for points in range(0, len(polyset[poly])):
                polyset[poly][points][0]=int(polyset[poly][points][0]-w_offset)
                polyset[poly][points][1]=int(polyset[poly][points][1]-h_offset)
        polys[name] = polyset
    return polys

def get_poly_vianum(infile, args):
    clipsize = args.load_size
    dtype = 0
    gdsii = gdspy.GdsLibrary(unit=1e-9)
    gdsii.read_gds(infile,units='convert')
    cell = gdsii.top_level()[0]
    bbox = cell.get_bounding_box()
    width = int((bbox[1,0]-bbox[0,0]))
    height= int((bbox[1,1]-bbox[0,1]))
    w_offset = int(bbox[0,0] - (clipsize-width)/2)
    h_offset = int(bbox[0,1] - (clipsize-height)/2)
    polys = {}
    try:
        polyset = cell.get_polygons(by_spec=True)[(LAYERS['design'],dtype)]
    except:
        print('layer via: 0 not found, skipping...')
        return None
    return len(polyset)

# def gds2img(Infolder, Infile, ImgOut):
#     GdsIn = os.path.join(Infolder, Infile)
#     gdsii   = gdspy.GdsLibrary(unit=1e-9)
#     gdsii.read_gds(GdsIn,units='convert')
#     cell    = gdsii.top_level()[0]
#     bbox    = cell.get_bounding_box()
#     opt_space=40 #Leave space at border in case of edge correction

#     width = int((bbox[1,0]-bbox[0,0]))
#     height= int((bbox[1,1]-bbox[0,1]))
#     w_offset = int(bbox[0,0] - (clipsize-width)/2)
#     h_offset = int(bbox[0,1] - (clipsize-height)/2)


#     sellayer = [DESIGN_LAYER, OPC_LAYER, SRAF_LAYER] #Layer Number
#     dtype = 0  #Layout Data Type
#     polygon  = []
#     im = Image.new("RGB", (clipsize, clipsize))
#     draw = ImageDraw.Draw(im)
#     token = 1
#     for i in range(len(sellayer)):
#         try:
#             polyset = cell.get_polygons(by_spec=True)[(sellayer[i],dtype)]
#         except:
#             token=0
#             # print("Layer not found, skipping...")
#             break
#         for poly in range(0, len(polyset)):
#             for points in range(0, len(polyset[poly])):
#                 polyset[poly][points][0]=int(polyset[poly][points][0]-w_offset)
#                 polyset[poly][points][1]=int(polyset[poly][points][1]-h_offset)
#         for j in range(0, len(polyset)):
#             tmp = tuple(map(tuple, polyset[j]))
#             if sellayer[i] == DESIGN_LAYER:
#                 draw.polygon(tmp, fill=(255, 0, 0))
#             if sellayer[i] == OPC_LAYER:
#                 draw.polygon(tmp, fill=(0, 255, 0))
#             if sellayer[i] == SRAF_LAYER:
#                 draw.polygon(tmp, fill=(0, 0, 255))
#     if token == 1:
#         filename = Infile+".png"
#         outpath  = os.path.join(ImgOut,filename)
#         im.save(outpath)



# # Infolder = sys.argv[1]
# # Outfolder= sys.argv[2]

# # Infolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_sample')
# Infolder = '/Users/dekura/Desktop/opc/datasets/lccout/gds'
# # Infolder = '/Users/dekura/Desktop/opc/design-april'
# # Outfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'test_sample_output/mask_sraf')
# Outfolder = '/Users/dekura/Desktop/opc/datasets/lccout/maskg_rgb'


# if not os.path.isdir(Outfolder):
#     os.mkdir(Outfolder)

# for dirname, dirnames, filenames in os.walk(Infolder):
#     bar=Bar("Converting GDSII to Image", max=len(filenames))
#     for f in range(0, len(filenames)):
#         try:
#             gds2img(Infolder, filenames[f], Outfolder)
#         except:
#             bar.next()
#             print('error')
#             continue
#         bar.next()
# bar.finish()