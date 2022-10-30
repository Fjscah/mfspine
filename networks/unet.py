import sys
from typing import Union

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import TruncatedNormal, VarianceScaling
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Conv3D,
                                     Conv3DTranspose, Dropout, MaxPool2D,
                                     MaxPool3D, MaxPooling2D, Softmax,
                                     UpSampling2D, concatenate,UpSampling3D)
from tensorflow.keras.regularizers import L2
from tensorflow.python.keras.layers import PReLU
import os
import csv
import h5py
import math

import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa

from skimage import transform
from skimage import exposure
from skimage.exposure import equalize_hist

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
# from dataloader import SparseDataGenerator, SparseTiffDataGenerator
# from utils import SampleImageCallback
# from utils import weighted_categorical_crossentropy
# from utils import dice_coefficient
sys.path.append(".")
import numpy as np
from utils.yaml_config import YAMLConfig, default_configuration


class UNet3D(Model):# no test
    def setting(self, configuration: YAMLConfig = None):

        if configuration == None:
            configuration = default_configuration
        self.configuration = configuration
        dict_a = self.configuration.config["Path"]
        dict_b = self.configuration.config["Network"]
        dict_c = self.configuration.config["Data"]
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        if self.kernel_regularizer == "L2":
            self.reg = L2(l2=self.reg_factor)
        else:
            self.reg = None
        #dropout
        self.dropout = Dropout(rate=self.droput_probability)
        # #left layer
        # self.layers_down=[] # for for downsample
        # self.layers_dtile=[] # fot concate

        # #right layer
        # self.layers_utile=[] # for for downsample
        # self.layers_up=[] # for for downsample

        # #pool layer
        # self.layers_upool=[] # fot upsample
        # self.layers_dpool=[]

    def __init__(self, configuration: YAMLConfig = None):
        super().__init__()
        self.setting(configuration)
        shallow_n_nstride = (1, 2, 2)
        self.norm = tf.keras.layers.LayerNormalization()
        
        self.conv11 = Conv3D(16, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        self.conv12 = Conv3D(32, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        self.pool1 = MaxPool3D(shallow_n_nstride, strides=shallow_n_nstride, name='max_1')

        self.conv21 = Conv3D(32, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        self.conv22 = Conv3D(64, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        self.pool2 = MaxPool3D(shallow_n_nstride, strides=shallow_n_nstride, name='max_2')
        self.drop2 = Dropout(0.25)#(conv5)
        



        # Level bottom
        self.conv_bottom1 = Conv3D(64, (3, 3, 3),kernel_regularizer=self.reg,name='conv_bt1',activation='relu',padding="same")
        self.conv_bottom2 = Conv3D(128, (3, 3, 3),kernel_regularizer=self.reg,name='conv_bt2',activation='relu',padding="same")
        self.drop_bottom = Dropout(0.5)#(conv5)



        self.up2 = UpSampling3D(size=shallow_n_nstride,name='upsample_2')
        self.conv23 = Conv3D(64, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        self.conv24 = Conv3D(64, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")

        self.up1 = UpSampling3D(size=shallow_n_nstride,name='upsample_1')# '_TupleWrapper' object is not callable ,please delete comma
        self.conv13 = Conv3D(32, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        self.conv14 = Conv3D(32, (3, 3, 3),kernel_regularizer=self.reg,activation='relu',padding="same")
        

        #Level out
        self.conv_out = Conv3D(self.num_classes,3,activation='softmax',padding='same',name='conv_out',kernel_initializer='he_normal')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # out = (inputs - tf.reduce_min(inputs)) / (
        #     0.8 * tf.reduce_max(inputs) - tf.reduce_min(inputs))

        out = self.conv11(inputs)
        o1 = self.conv12(out)
        out = self.pool1(o1)
        # print(out.shape)

        
        
        out = self.conv21(out)
        self.drop2(out)
        o2=self.conv22(out)
        out=self.pool2(o2)
        # print(out.shape)

        
        out=self.conv_bottom1(out)
        out=self.conv_bottom2(out)
        self.drop_bottom(out)
        # print(out.shape)

        
        out=self.up2(out)
        out=concatenate([out,o2])
        out=self.conv23(out)
        out=self.conv24(out)
        # print(out.shape)
  
        
        out=self.up1(out)
        out=concatenate([out,o1])
        out=self.conv13(out)
        out=self.conv14(out)
        # print(out.shape)
        

        out = self.conv_out(out)
        # print(out.shape)

        return out


class UNet2D(Model):  
    def setting(self, configuration: YAMLConfig = None):
        if configuration == None:
            configuration = default_configuration
        self.configuration = configuration
        dict_a = self.configuration.config["Path"]
        dict_b = self.configuration.config["Network"]
        dict_c = self.configuration.config["Data"]
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        if self.kernel_regularizer == "L2":
            self.reg = L2(l2=self.reg_factor)
        else:
            self.reg = None
        #dropout
        self.dropout = Dropout(rate=self.droput_probability)
        
        #left layer
        self.layers_down = []  # for for downsample
        self.layers_dtile = []  # fot concate

        #right layer
        self.layers_utile = []  # for for downsample
        self.layers_up = []  # for for downsample

        #pool layer
        self.layers_upool = []  # fot upsample
        self.layers_dpool = []

    def __init__(self, configuration: YAMLConfig = None):
        super().__init__()
        self.setting(configuration)
        self.norm = tf.keras.layers.BatchNormalization()
        self.norm2 = ReLU()
        # inputs = Input((self.img_rows, self.img_cols, 1))

        self.conv11 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(inputs)
        self.conv12 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))#(conv1)

        self.conv21 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(pool1)
        self.conv22 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv2)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))#(conv2)
       

        self.conv31 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(pool2)
        self.conv32 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv3)
        self.pool3 = MaxPooling2D(pool_size=(2, 2))#(conv3)
        self.drop3 = Dropout(0.25)#(conv5)
        
        self.conv41 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(pool4)
        self.conv42 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv5)
        self.drop4 = Dropout(0.5)#(conv5)
        
        self.up3=UpSampling2D(size=(2, 2))
        self.conv33 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(pool1)
        self.conv34 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv2)
        
        self.up2=UpSampling2D(size=(2, 2))
        self.conv23 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(pool1)
        self.conv24 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv2)
        
        self.up1=UpSampling2D(size=(2, 2))
        self.conv13 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(pool1)
        self.conv14 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')#(conv2)
                              

        self.convout = Conv2D(self.num_classes, 3, activation='softmax', padding='same', kernel_initializer='he_normal')#(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # self.classout = Softmax()#(conv9)
    @property
    def pool_num(self):
        return 3
 
    def call(self, inputs, training=None, mask=None):
        
        #out = tf.sqrt( tf.math.reduce_sum(tf.image.sobel_edges(inputs) ** 2, axis = -1) )
        # out=normalize(x,1,99.8,axis=axis_norm)
      
        out=self.norm(inputs)
        out=self.norm2(out)
        
        out = self.conv11(out)
        o1=self.conv12(out)
        out=self.pool1(o1)
        
        out = self.conv21(out)
        o2=self.conv22(out)
        out=self.pool2(o2)
        
        out = self.conv31(out)
        out=self.drop3(out)
        o3=self.conv32(out)
        out=self.pool3(o3)
        
        out = self.conv41(out)
        out=self.conv42(out)
        out=self.drop4(out)
        
        out=self.up3(out)
        out=concatenate([out,o3])
        out=self.conv33(out)
        out=self.conv34(out)
        
        out=self.up2(out)
        out=concatenate([out,o2])
        out=self.conv23(out)
        out=self.conv24(out)
        
        out=self.up1(out)
        out=concatenate([out,o1])
        out=self.conv13(out)
        out=self.conv14(out)
          
        out = self.convout(out)
          
        # out = self.convout(out)
        #out=self.classout(out)
        return out   
    
    def predict_single_img(self,img):
        """
        imgs: shape (x,y,c=1,None)
             shape(z,x,y,c=1,None)
        """
        model=self
        cm=2**model.pool_num
        ndim=img.ndim
        shape=img.shape
        padd=[(0,(cm-s%cm)%8) for s in img.shape]
        if ndim==3:
            padd[0]=(0,0)
        img=np.pad(img,padd)
        img=np.expand_dims(img,axis=-1).astype(np.float32)
        if ndim==2:
            img=np.expand_dims(img,axis=0).astype(np.float32)
        imgns=[]
        spineprs=[]
        denprs=[]
        # print(img.shape)
        for im in img:
            im=np.expand_dims(im,axis=0).astype(np.float32)
            ppr=model.predict(im)[0]
            # pr = ppr.argmax(axis=-1)
            imgn = ppr.argmax(axis=-1)
            spineprs.append(ppr[...,2])
            denprs.append(ppr[...,1])
            imgns.append(imgn)
        imgns=np.array(imgns) # label 0 1 2 mask
        imgns=np.squeeze(imgns)
        spineprs=np.array(spineprs)
        spineprs=np.squeeze(spineprs)
        denprs=np.array(denprs)
        denprs=np.squeeze(denprs)
        # print(shape,imgns.shape)
        
        obj=[slice(0,s) for s in shape ]
        #print(imgns.shape,spineprs.shape,denprs.shape,obj)
        return imgns[obj],spineprs[obj],denprs[obj]
      

        
        
    
    def predict_time_imgs(self,imgs):
        # print("img shape : ",img.shape)
        masks=[]
        spine_prs=[]
        den_prs=[]
        n_frames=imgs.shape[0]
        for i in tqdm(range(n_frames), desc='Processing'):
            img=imgs[i]
            mask,spineprs,denprs=self.predict_single_img(img)
            masks.append(mask)
            spine_prs.append(spineprs)
            den_prs.append(denprs)
        masks=np.array(masks)
        masks=np.squeeze(masks)    
        spine_prs=np.array(spine_prs)
        spine_prs=np.squeeze(spine_prs) 
        den_prs=np.array(den_prs)
        den_prs=np.squeeze(den_prs) 
        return masks,spine_prs,den_prs
        
        
        
        
    """        
        for layern in range(1, self.layer_num):
            convd1_n = Conv2D(
                self.base_filter * layern * 2, (3, 3),
                name='conv_d1_' + str(layern),
                padding="same",
                kernel_regularizer=self.reg,activation=self.activation_function,kernel_initializer=self.kernel_init)
            convd2_n = Conv2D(
                self.base_filter * layern * 2, (3, 3),
                name='conv_d2_' + str(layern),
                padding="same",
                kernel_regularizer=self.reg,activation=self.activation_function,kernel_initializer=self.kernel_init)

            convu1_n = Conv2D(
                self.base_filter * layern * 2, (3, 3),
                name='conv_u1_' + str(layern),
                padding="same",
                kernel_regularizer=self.reg,activation=self.activation_function,kernel_initializer=self.kernel_init)
            convu2_n = Conv2D(
                self.base_filter * layern * 2, (3, 3),
                name='conv_u2_' + str(layern),
                padding="same",
                kernel_regularizer=self.reg,activation=self.activation_function,kernel_initializer=self.kernel_init)

            max_n = MaxPool2D(2, 2, name='max_' + str(layern))
            up_n = Conv2DTranspose(
                self.base_filter * layern * 2,
                (2, 2), strides=(2, 2),
                name='upsample_' + str(layern),
                padding="same")

            self.layers_down.append(convd1_n)
            self.layers_dtile.append(convd2_n)

            self.layers_utile.append(convu1_n)
            self.layers_up.append(convu2_n)

            self.layers_dpool.append(max_n)
            self.layers_upool.append(up_n)

        # Level bottom
        self.conv_bottom1 = Conv2D(
            self.base_filter * (self.layer_num - 1) * 2, (3, 3),
            name='conv_bt1',
            padding="same",kernel_regularizer=self.reg,activation=self.activation_function,kernel_initializer=self.kernel_init)
        self.conv_bottom2 = Conv2D(
            self.base_filter * (self.layer_num - 2) * 2, (3, 3),
            name='conv_bt2',
            padding="same",kernel_regularizer=self.reg,activation=self.activation_function,kernel_initializer=self.kernel_init)
        #Level out
        self.conv_out = Conv2D(
            self.num_classes,
            1,
            1,
            padding='same',
            name='conv_out',
            dtype='float32',activation="softmax")
    """
 

#https://github.com/krentzd/sparse-unet
def down_block_2D(input_tensor, filters):

    x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def up_block_2D(input_tensor, concat_layer, filters):

    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)

    x = Concatenate()([x, concat_layer])

    x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def sunet_2D(shape,classnum=3, filters=32):

    input_tensor = Input(shape, name='img')


    d1 = down_block_2D(input_tensor, filters=filters)
    p1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d1)
    d2 = down_block_2D(p1, filters=filters*2)
    p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d2)
    d3 = down_block_2D(p2, filters=filters*4)
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d3)

    d4 = down_block_2D(p3, filters=filters*8)

    u1 = up_block_2D(d4, d3, filters=filters*4)
    u2 = up_block_2D(u1, d2, filters=filters*2)
    u3 = up_block_2D(u2, d1, filters=filters)

    # Returns one-hot-encoded semantic segmentation mask where 0 is bakcground, 1 is mito and 2 is None (weight zero)
    output_tensor = Conv2D(filters=classnum, kernel_size=(1,1), activation='softmax')(u3)

    return Model(inputs=[input_tensor], outputs=[output_tensor])


if __name__ =="__main__":
    # model=UNet3D()
    # model.build(input_shape=(4,10,128,128,1))
    # model.summary()
    # model=UNet2D()
    # model.build(input_shape=(4,256,256,1))
    # model.summary()
    # model.build(input_shape=(4,512,512,1))
    # model.summary()
    model=sunet_2D((256,256,1))
    model.summary()
    model=sunet_2D((512,512,1))
    model.summary()
    