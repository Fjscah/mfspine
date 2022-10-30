#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2022/04/26 19:28:21
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   GNU, LNP2 group
@Desc    :   load dataset from folder
'''

# here put the import lib
from glob import glob

import sys

import numpy as np
import tensorflow as tf

sys.path.append(".")
import math
import os
import shutil
from functools import partial
from random import shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
from tensorflow import keras

from utils import file_base, yaml_config
from utils.basic_wrap import timing
from utils.yaml_config import YAMLConfig, default_configuration


# from data_loader.preprocessing import PaddedStackPatcher
def show_two(im1,im2):
    fig,axes=plt.subplots(1,2,sharex=True,sharey=True)
    axes[0].imshow(im1,extent=[0, im1.shape[0], 0, im1.shape[0]])
    axes[1].imshow(im2,extent=[0, im2.shape[0], 0, im2.shape[0]])
    plt.show()

def get_from_numpy(data,label):
    return tf.data.Dataset.from_tensor_slices(data,label)



def load_img(filename=""):
    if os.path.exists(filename):
        return imread(filename)
    else:
        raise FileNotFoundError("ERR file "+filename+" not exist \n")

def load_datate_from_folder(configuration: YAMLConfig):
    ptr,pva,pte=configuration.get_entry(['Data', 'partion'])
    imfiles=file_base.file_list(configuration.get_entry(['Path', 'data_path']))
    lafiles=file_base.file_list(configuration.get_entry(['Path', 'label_path']))
    filepairs=file_base.pair_files(imfiles,lafiles,configuration.get_entry(['Path', 'label_suffix']))
    shuffle(filepairs)
    length=len(filepairs)

    trdatast,vadataset,tedataset=CusDataset(configuration),CusDataset(configuration),CusDataset(configuration)
    trdatast.set_filelist(filepairs[:int(length*ptr)]) 
    vadataset.set_filelist(filepairs[int(length*ptr):int(length*(ptr+pva))]) 
    tedataset.set_filelist(filepairs[int(length*(ptr+pva)):]) 
    return trdatast,vadataset,tedataset
    
     

class CusDataset(keras.utils.Sequence):
    def setting(self, configuration: YAMLConfig = None):
        if configuration == None:
            configuration = default_configuration
        self.configuration=configuration  
        dict_a=self.configuration.config["Path"]
        dict_b=self.configuration.config["Data"]
        dict_c=self.configuration.config["Training"]
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.initial_filelist()
        self.inital_shape()
        
        
    def inital_shape(self):
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            self.imshape=tf.stack([nz, w, w, 1])
            self.labshape=tf.stack([nz, w, w, self.num_classes])
        else:
            self.imshape=tf.stack([ w, w, 1])    
            self.labshape=tf.stack([ w, w, self.num_classes])    
    def initial_filelist(self):
        imfiles=file_base.file_list(self.data_path)
        lafiles=file_base.file_list(self.label_path)
        self.filepairs=file_base.pair_files(imfiles,lafiles,self.label_suffix)
        self.filenum=len(self.filepairs)
        self.on_epoch_begin()
    def set_filelist(self,filelist):
        self.filepairs=filelist
        self.filenum=len(self.filepairs)
        self.on_epoch_begin()
        
        

    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)

    def __len__(self):
        return math.ceil(self.filenum / float(self.batch_size))

    def __getitem__(self, index):
        images  = []
        targets = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.filenum
            imgf=self.filepairs[i][0]
            labf=self.filepairs[i][1]
            im=load_img(imgf)
            lab=load_img(labf)
            lab=lab.astype(np.int64)
            #print(np.unique(lab))
            lab=tf.keras.utils.to_categorical(lab, num_classes=self.num_classes)
            im=im.astype(np.float32)
            #print(lab.shape)
            lab=np.reshape(lab,self.labshape)
            im=np.reshape(im,self.imshape)
            
            
            images.append(im)
            targets.append(lab)

        images  = np.array(images)
        targets = np.array(targets)
        return images, targets

    # def generate(self):
    #     i = 0
    #     while True:
    #         images  = []
    #         targets = []
    #         for b in range(self.batch_size):
    #             if i==0:
    #                 np.random.shuffle(self.annotation_lines)
    #             name        = self.annotation_lines[i].split()[0]
    #             #-------------------------------#
    #             #   从文件中读取图像
    #             #-------------------------------#
    #             jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
    #             png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
    #             #-------------------------------#
    #             #   数据增强
    #             #-------------------------------#
    #             jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
    #             jpg         = preprocess_input(np.array(jpg, np.float64))
    #             png         = np.array(png)
    #             png[png >= self.num_classes] = self.num_classes
    #             #-------------------------------------------------------#
    #             #   转化成one_hot的形式
    #             #   在这里需要+1是因为voc数据集有些标签具有白边部分
    #             #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
    #             #-------------------------------------------------------#
    #             seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
    #             seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

    #             images.append(jpg)
    #             targets.append(seg_labels)
    #             i           = (i + 1) % self.length
                
    #         images  = np.array(images)
    #         targets = np.array(targets)
    #         yield images, targets
            
    # def rand(self, a=0, b=1):
    #     return np.random.rand() * (b - a) + a

    # def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    #     image = cvtColor(image)
    #     label = Image.fromarray(np.array(label))
    #     h, w = input_shape

    #     if not random:
    #         iw, ih  = image.size
    #         scale   = min(w/iw, h/ih)
    #         nw      = int(iw*scale)
    #         nh      = int(ih*scale)

    #         image       = image.resize((nw,nh), Image.BICUBIC)
    #         new_image   = Image.new('RGB', [w, h], (128,128,128))
    #         new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    #         label       = label.resize((nw,nh), Image.NEAREST)
    #         new_label   = Image.new('L', [w, h], (0))
    #         new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    #         return new_image, new_label

    #     # resize image
    #     rand_jit1 = self.rand(1-jitter,1+jitter)
    #     rand_jit2 = self.rand(1-jitter,1+jitter)
    #     new_ar = w/h * rand_jit1/rand_jit2

    #     scale = self.rand(0.25, 2)
    #     if new_ar < 1:
    #         nh = int(scale*h)
    #         nw = int(nh*new_ar)
    #     else:
    #         nw = int(scale*w)
    #         nh = int(nw/new_ar)

    #     image = image.resize((nw,nh), Image.BICUBIC)
    #     label = label.resize((nw,nh), Image.NEAREST)
        
    #     flip = self.rand()<.5
    #     if flip: 
    #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #         label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
    #     # place image
    #     dx = int(self.rand(0, w-nw))
    #     dy = int(self.rand(0, h-nh))
    #     new_image = Image.new('RGB', (w,h), (128,128,128))
    #     new_label = Image.new('L', (w,h), (0))
    #     new_image.paste(image, (dx, dy))
    #     new_label.paste(label, (dx, dy))
    #     image = new_image
    #     label = new_label

    #     # distort image
    #     hue = self.rand(-hue, hue)
    #     sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
    #     val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
    #     x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    #     x[..., 0] += hue*360
    #     x[..., 0][x[..., 0]>1] -= 1
    #     x[..., 0][x[..., 0]<0] += 1
    #     x[..., 1] *= sat
    #     x[..., 2] *= val
    #     x[x[:,:, 0]>360, 0] = 360
    #     x[:, :, 1:][x[:, :, 1:]>1] = 1
    #     x[x<0] = 0
    #     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
    #     return image_data,label

    def on_epoch_begin(self):
        shuffle(self.filepairs)



class UnetDataset(keras.utils.Sequence):
    def __init__(self, datafolder, input_shape,suffix, batch_size, num_classes):
        self.imgfiles = glob(datafolder+'\\img\\*.tif')
        self.labelfiles = glob(datafolder+'\\label\\*.tif')
        self.pairs=file_base.pair_files(self.imgfiles,self.labelfiles,suffix=suffix)
        
        self.length             = len(self.imgfiles)
        print("======="+self.__class__.__name__,"======\nlength :",len(self.labelfiles),len(self.imgfiles),
              "\nload folder:",datafolder,"\n=========================")
        # print(self.pairs[0])
        self.input_shape        = input_shape # img [z] x y c shape
        self.batch_size         = batch_size
        self.num_classes        = num_classes


    def __len__(self):
        return math.ceil(self.length/ float(self.batch_size))

    def __getitem__(self, index):
        images  = []
        targets = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            
            #-------------------------------#
            #   load img from file
            #-------------------------------#
            im         = imread(self.pairs[i][0])
            lab         = imread(self.pairs[i][1])
            #-------------------------------#
            #   数据增强
            #-------------------------------#
            #jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
            #jpg         = preprocess_input(np.array(jpg, np.float64))
            #png         = np.array(png)
            #png[png >= self.num_classes] = self.num_classes
            #-------------------------------------------------------#
            # trans to one_hot的form
            # (x,y)->(x,y,code)
            # eg 1-> 010 ;  2-> 100
            #-------------------------------------------------------#
            im=im.reshape(self.input_shape)
            
            seg_labels  = np.eye(self.num_classes )[lab]
            #seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

            images.append(im)
            targets.append(seg_labels)

        images  = np.array(images,dtype=np.float32)
        targets = np.array(targets,dtype=np.float32)
        
        return images, targets

    # def generate(self):
    #     i = 0
    #     while True:
    #         images  = []
    #         targets = []
    #         for b in range(self.batch_size):
    #             if i==0:
    #                 np.random.shuffle(self.annotation_lines)
    #             name        = self.annotation_lines[i].split()[0]
    #             #-------------------------------#
    #             #   从文件中读取图像
    #             #-------------------------------#
    #             jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
    #             png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
    #             #-------------------------------#
    #             #   数据增强
    #             #-------------------------------#
    #             jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
    #             jpg         = preprocess_input(np.array(jpg, np.float64))
    #             png         = np.array(png)
    #             png[png >= self.num_classes] = self.num_classes
    #             #-------------------------------------------------------#
    #             #   转化成one_hot的形式
    #             #   在这里需要+1是因为voc数据集有些标签具有白边部分
    #             #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
    #             #-------------------------------------------------------#
    #             seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
    #             seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

    #             images.append(jpg)
    #             targets.append(seg_labels)
    #             i           = (i + 1) % self.length
                
    #         images  = np.array(images)
    #         targets = np.array(targets)
    #         yield images, targets
            
    def on_epoch_end(self):
        #shuffle file list
        rng = np.random.RandomState(np.random.randint(100))
        inds = rng.permutation(self.length)
        self.pairs=[self.pairs[i] for i in inds]
        
         
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a




#==↓==↓==↓==↓==↓==↓== load data from folder :train test validation ==↓==↓==↓==↓==↓==↓#
def load_image(img_path="",note="seg"):
    print(img_path.numpy())
    img= imread(img_path)
    img=tf.image.resize(img,img.shape)
    ldir,lname,lsufix=file_base.split_img_path(img_path)
    ldir=file_base.get_parent_dir(ldir,1)
    labelfile=os.path.join(ldir,"label",lname+note+lsufix)
    lab=imread(labelfile)
    return (img,lab)



class DataLoader:
    def setting(self, configuration: YAMLConfig = None):
        if configuration == None:
            configuration = default_configuration
        self.configuration=configuration  
        dict_a=self.configuration.config["Path"]
        dict_b=self.configuration.config["Data"]
        dict_c=self.configuration.config["Training"]
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.initial_filelist()
        self.inital_shape()
    def inital_shape(self):
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            self.imshape=tf.stack([nz, w, w, 1])
            self.labshape=tf.stack([nz, w, w, self.num_classes])
        else:
            self.imshape=tf.stack([ w, w, 1])    
            self.labshape=tf.stack([ w, w, self.num_classes])     
    def initial_filelist(self):
        imfiles=file_base.file_list(self.data_path)
        lafiles=file_base.file_list(self.label_path)
        self.filepairs=file_base.pair_files(imfiles,lafiles,self.label_suffix)
        self.filenum=len(self.filepairs)
        shuffle(self.filepairs)
        
    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)
    def get_dataset(self):
        #使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
        Train_path=self.Train_path
        trpath=Train_path+"train/img/"+"*"+self.filetype
        tepath=Train_path+"test/img/"+"*"+self.filetype
        vapath=Train_path+"valid/img/"+"*"+self.filetype
        batch_size=self.batch_size
        print(trpath,tepath,vapath)
        loadim = partial(load_image, note=self.label_suffix)
        ds_train = tf.data.Dataset.list_files(trpath) \
                .map(loadim) \
                .shuffle(buffer_size = 1000).batch(batch_size) 
                 

        ds_test = tf.data.Dataset.list_files(tepath) \
                .map(loadim) \
                .batch(batch_size) 
                 
        ds_valid = tf.data.Dataset.list_files(vapath) \
                .map(loadim) \
                .batch(batch_size) 
                
        return ds_train,ds_valid,ds_test
    def show_data(self,ds):
        for i,(img,label) in enumerate(ds.unbatch().take(9)):
            ax=plt.subplot(3,3,i+1)
            ax.imshow(img.numpy())
            ax.set_title("label = %d"%label)
            ax.set_xticks([])
            ax.set_yticks([]) 
        plt.show()
    def split_data(self):
        Train_path=self.Train_path
        split_partion=self.partion # train,valid,test+=1
        accu_partion=[split_partion[0],split_partion[0]+split_partion[1],1]
        path_ts=["train","valid","test"]

        length=len(self.filepairs)
        self.filenum=len(self.filepairs)
        t=0
        for path_type,p in zip(path_ts,accu_partion):
            files=self.filepairs[t:int(p*length)]
            t=int(p*length)
            imgdir=os.path.join(Train_path,path_type+"/img")
            labdir=os.path.join(Train_path,path_type+"/label")
            file_base.create_dir(imgdir)
            file_base.create_dir(labdir)
            for imfile,labfile in files:
                _,name,suffix=file_base.split_filename(imfile)
                shutil.copyfile(imfile, os.path.join(imgdir,name+suffix))
                _,name,suffix=file_base.split_filename(labfile)
                shutil.copyfile(labfile, os.path.join(labdir,name+suffix))


class SliceLoader:
    def setting(self, configuration: YAMLConfig = None):
        if configuration == None:
            configuration = default_configuration
        self.configuration=configuration  
        dict_a=self.configuration.config["Path"]
        dict_b=self.configuration.config["Data"]
        dict_c=self.configuration.config["Training"]
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.initial_filelist()
        self.inital_shape()
    def inital_shape(self):
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            self.imshape=tf.stack([nz, w, w, 1])
            self.labshape=tf.stack([nz, w, w, self.num_classes])
        else:
            self.imshape=tf.stack([ w, w, 1])    
            self.labshape=tf.stack([ w, w, self.num_classes])     
    def initial_filelist(self):
        imfiles=file_base.file_list(self.data_path)
        lafiles=file_base.file_list(self.label_path)
        self.filepairs=file_base.pair_files(imfiles,lafiles,self.label_suffix)
        self.filenum=len(self.filepairs)
        shuffle(self.filepairs)
        
    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)
    
    def img_shape(self):
        if "z" in self.axes:
            self.shape=[self.input_sizez,self.input_sizexy,self.input_sizexy]
        else:
            self.shape=[self.input_sizexy,self.input_sizexy]
    @staticmethod
    def load_ds_dir(ddir,note="seg",suffix=".tif"):
        impath=os.path.join(ddir,"img")
        lapath=os.path.join(ddir,"label")
        imfiles=file_base.file_list(impath,suffix)
        lafiles=file_base.file_list(lapath,suffix)
        filepairs=file_base.pair_files(imfiles,lafiles,note)
        ims=[]
        las=[]
        for imfile,lafile in filepairs:
            im=imread(imfile)
            la=imread(lafile)
            # print(im.shape)
            im=im.reshape(list(im.shape)+[1]).astype(np.float32)
            la=la.reshape(list(im.shape)+[1]).astype(np.float32)
            # print(im.shape)
            ims.append(im)
            las.append(la)
        ims=np.array(ims)
        print("Data shape",ims.shape)
        print("======================")
        return  ims,np.array(las)
    @timing    
    def get_dataset(self):
        #使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
        Train_path=self.Train_path
        trpath=os.path.join(Train_path,"train") 
        tepath=os.path.join(Train_path,"test") 
        vapath=os.path.join(Train_path,"valid") 
        batch_size=self.batch_size
        note=self.label_suffix
        filetype=self.filetype
        print("Dataset Folder:")
        print("\n".join([trpath,tepath,vapath]))
        print("=============================================")
        ds_train = SliceLoader.load_ds_dir(trpath,note,filetype) 
        ds_test = SliceLoader.load_ds_dir(tepath,note,filetype) 
        ds_valid = SliceLoader.load_ds_dir(vapath,note,filetype) 

        return ds_train,ds_valid,ds_test
    def split_data(self):
        Train_path=self.Train_path
        split_partion=self.partion # train,valid,test+=1
        accu_partion=[split_partion[0],split_partion[0]+split_partion[1],1]
        path_ts=["train","valid","test"]

        length=len(self.filepairs)
        self.filenum=len(self.filepairs)
        t=0
        for path_type,p in zip(path_ts,accu_partion):
            files=self.filepairs[t:int(p*length)]
            t=int(p*length)
            imgdir=os.path.join(Train_path,path_type+"/img")
            labdir=os.path.join(Train_path,path_type+"/label")
            file_base.remove_dir(imgdir)
            file_base.create_dir(imgdir)
            file_base.remove_dir(labdir)
            file_base.create_dir(labdir)
            for imfile,labfile in files:
                _,name,suffix=file_base.split_filename(imfile)
                shutil.copyfile(imfile, os.path.join(imgdir,name+suffix))
                _,name,suffix=file_base.split_filename(labfile)
                shutil.copyfile(labfile, os.path.join(labdir,name+suffix))
            print("complete folder : ", os.path.join(Train_path,path_type))
    @staticmethod
    def show_data(ds):
        def showtwo(im1,im2):
            fig,axes=plt.subplots(1,2,sharex=True,sharey=True)
            print(im1.shape)
            axes[0].imshow(im1[:,:,0])
            axes[1].imshow(im2[:,:,0])
            plt.show()
        imgs,labs=ds
        for i,(img,label) in enumerate(zip(imgs[:9],labs[0:9])):
            showtwo(img,label)         
if __name__=="__main__":
    dataloader=SliceLoader()
    # tf.executing_eagerly()
    #trds,_,_=dataloader.get_dataset()
    #SliceLoader.show_data(trds)
    dataloader.split_data()             

        