import numpy as np
from skimage.util.shape import view_as_windows,view_as_blocks
from skimage.util import montage
from skimage.filters.thresholding import (threshold_isodata, threshold_li,
                                          threshold_mean, threshold_minimum,
                                          threshold_multiotsu,
                                          threshold_niblack, threshold_otsu,
                                          threshold_triangle, threshold_yen,threshold_local)
import sys
sys.path.append(".")
from networks import unet
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def showtwo(im1,im2):
    fig,axes=plt.subplots(1,2,sharex=True,sharey=True)
    print(im1.shape)
    axes[0].imshow(im1)
    axes[1].imshow(im2)
    plt.show()
import napari
num_classes=3
from utils.file_base import file_list
import colorsys
from PIL import Image
from skimage.segmentation import watershed,morphological_chan_vese

def split_patches(img,cropsize):
    imgpad=np.pad(img,[(0,s%c) for s,c in zip(img.shape,cropsize)])
    splitshape=[s//c for s,c in zip(imgpad.shape,cropsize)]
    B = view_as_blocks(imgpad, cropsize)
    print(B.shape)
    B=B.reshape(-1,*cropsize)
    return B,splitshape
def merge_patches(B,splitshape):
    arr_out = montage(B,grid_shape=splitshape)
    return arr_out
def predict_dir(modelpath=r"D:\spine\spinesoftware\myspine\models\M2d_den\modelep010-loss0.023.h5"):
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    model=unet.UNet2D()
    model.build(input_shape =(4,256,256,1))
    model.load_weights(modelpath)
#     model=tf.keras.models.load_model(
#     r"D:\spine\spinesoftware\myspine\models\M2d_den\model", custom_objects=None, compile=True, options=None
# )
    imgdir=r"C:\Users\ZLY\Desktop\Train\4D\deconvole_2D_one"


        
    imglist=file_list(imgdir)
    for imgf in imglist:
        print(imgf)
        img=imread(imgf)
        imgs,splitshape=split_patches(img,cropsize=(256,256))
        imgsn=[]
        for im in imgs:
            im=im.reshape(1,256,256,1).astype(np.float32)
            ppr=model.predict(im)[0]
            pr = ppr.argmax(axis=-1)
            imgsn.append(pr)
        imgn=merge_patches(imgsn,splitshape)
        imgn=imgn[0:img.shape[0],0:img.shape[1]]


        # res=np.max(out,axis=-1)
        showtwo(img,imgn)
    # print(out)
def predict_movie(imgf=r"D:\spine\Train\4D\deconvolve_2D\MAX_decon_20211111-24D.tif",
          modelpath=r"D:/spine/spinesoftware/myspine/models/M2d_den\modelep008-loss0.013.h5"):
    imgs=imread(imgf)
    model=unet.UNet2D()
#     model=tf.keras.models.load_model(
#     r"D:\spine\spinesoftware\myspine\models\M2d_den\model", custom_objects=None, compile=True, options=None
# )
    model.build(input_shape =(4,512,512,1))
    model.load_weights(modelpath)
    print("img shape : ",imgs.shape)
    imgns=[]
    for img in imgs:
        img=img.reshape(1,512,512,1).astype(np.float32)
        print(img.shape)
        ppr=model.predict(img)[0]
        pr = ppr.argmax(axis=-1)
        imgn = ppr.argmax(axis=-1)
        # imgss,splitshape=split_patches(img,cropsize=(256,256))
        # imgsn=[]
        # for im in imgss:
        #     im=im.reshape(1,256,256,1).astype(np.float32)
        #     ppr=model.predict(im)[0]
        #     pr = ppr.argmax(axis=-1)
        #     imgsn.append(pr)
        # imgn=merge_patches(imgsn,splitshape)
        # imgn=imgn[0:img.shape[0],0:img.shape[1]]
        imgns.append(imgn)
    return imgns,imgs
def predict_4D(imgf=r"D:\data\Train\4D\deconvolve_4D\decon_20211111-24D.tif",
          modelpath=r"D:/spine/spinesoftware/myspine/models/M2d_seg\modelep104-loss0.047.h5"):
    imgss=imread(imgf)[:1]
    adth=imgss[0]>threshold_otsu(imgss[0])
    # adth=morphological_chan_vese(imgss[0],20,adth,1,1,2)
    model=unet.UNet2D()
#     model=tf.keras.models.load_model(
#     r"D:\spine\spinesoftware\myspine\models\M2d_den\model", custom_objects=None, compile=True, options=None
# )
    model.build(input_shape =(4,512,512,1))
    model.load_weights(modelpath)
    print("img shape : ",imgss.shape)
    imgnss=[]
    for imgs in imgss:
        
        imgns=[]
        prs=[]
        for img in imgs:
            img=img.reshape(1,512,512,1).astype(np.float32)
            ppr=model.predict(img)[0]
            imgn = ppr.argmax(axis=-1)
            prs.append(ppr[...,2])
            
            # imgss,splitshape=split_patches(img,cropsize=(256,256))
            # imgsn=[]
            # for im in imgss:
            #     im=im.reshape(1,256,256,1).astype(np.float32)
            #     ppr=model.predict(im)[0]
            #     pr = ppr.argmax(axis=-1)
            #     imgsn.append(pr)
            # imgn=merge_patches(imgsn,splitshape)
            # imgn=imgn[0:img.shape[0],0:img.shape[1]]
            imgns.append(imgn)
        imgns=np.array(imgns)
        mask=np.max(imgs,axis=0)[None,...]
        mask=mask.reshape(1,512,512,1).astype(np.float32)
        ppr=model.predict(mask)[0]
        mask = ppr.argmax(axis=-1)==1
        mask=np.tile(mask,(imgns.shape[0],1,1))
        imgns[(imgns==2)*mask]=0
    imgnss.append(imgns)    
    return imgnss,imgss,prs,adth,mask
a,b,c,d,e=predict_4D()
viewer=napari.Viewer()
viewer.add_image(np.array(b))
viewer.add_image(np.array(c))
viewer.add_image(np.array(e))
viewer.add_image(np.array(d*a))
viewer.add_labels(np.array(a))
napari.run()