
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

num_classes=3
from utils.file_base import file_list
import colorsys
from PIL import Image
hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
# model=unet.UNet2D()
# model.build(input_shape =(4,256,256,1))
model=unet.sunet_2D((256,256,1))
modelpath=r"D:\code\myspine\models\MM2d_seg\modelep138-loss0.092.h5"
model.load_weights(modelpath)
imgdir=r"D:\data\Train\2Dimg_seg\test\img"

#---------------------------------------------------#
#   创建一副新图，并根据每个像素点的种类赋予颜色
#---------------------------------------------------#
imglist=file_list(imgdir)
# imglist=[
#     r"D:\data\Train\2Dimg_seg\test\img\2.tif",
# r"D:\data\Train\2Dimg_seg\test\img\4.tif",
# r"D:\data\Train\2Dimg_seg\test\img\1.tif",
# ]
for img in imglist:
    print(img)
    im=imread(img)
    im=im.reshape(1,256,256,1).astype(np.float32)
    ppr=model.predict(im)[0]
    ppr.shape
    # plt.imshow(ppr[:,:,2])
    # plt.show()
    from csbdeep.utils import Path, normalize
    imm = normalize(im,1,99.8,axis=None)
    layer = tf.keras.layers.LayerNormalization(axis=[1,2,3])
    imm=layer(imm)
    #print(imm)
    pr = ppr.argmax(axis=-1)
    seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    
    for c in range(3):
        seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    #------------------------------------------------#
    #   将新图片转换成Image的形式
    #------------------------------------------------#
    image = Image.fromarray(np.uint8(seg_img))

    # res=np.max(out,axis=-1)
    showtwo(imm[0,:,:,0],ppr)
# print(out)
