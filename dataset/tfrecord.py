import os
import sys
from paddle import dtype
import tensorflow as tf
import nibabel as nib
import numpy as np

import numpy as np
import tensorflow as tf
from skimage.io import imread,imsave
import os
import sys
import tensorflow as tf 
sys.path.append(".")
from utils.yaml_config import YAMLConfig, default_configuration
import glob
from skimage.measure import label, regionprops, regionprops_table
from utils import file_base

import numpy as np
from sympy import im
import imgaug.augmenters as iaa
import imgaug as ia
# Standard scenario: You have N=16 RGB-images and additionally one segmentation
# map per image. You want to augment each image and its heatmaps identically.
import matplotlib.pyplot as plt
# from .segment import fill_hulls,resortseg
from .dataloader import load_img,show_two
from sklearn.model_selection import train_test_split
import tensorflow as tf
# import tensorflow_transform as tft

tf.compat.v1.enable_eager_execution()


def get_img_label(imdir,ladir,note="seg",pathonly=True):
    imdir=os.path.abspath(imdir)
    ladir=os.path.abspath(ladir)
    print(imdir,ladir,"\n")
    imfiles=file_base.file_list(imdir)
    lafiles=file_base.file_list(ladir)
    pairs=file_base.pair_files(imfiles,lafiles,note)
    print("pair number : ",len(pairs))
    if pathonly:
        return pairs
    images=[]
    segmaps=[]
    #print(imdir,"\n",ladir,imfiles,lafiles,pairs)
    
    for img,labelf in pairs:
        # print(img,"\t",labelf)
       
        im=load_img(img)
        lab=load_img(labelf)

        # im=im.swapaxes(0,2) #z to depth
        # lab=lab.swapaxes(0,2)
        #print(im.shape)
        
        images.append(im)
        segmaps.append(lab)
    return images,segmaps

def read_tfrecord(tfrecord_file,numclass,shape):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件
    #print("sahpe",shape)
    feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'img': tf.io.VarLenFeature(tf.int64),
    #'shape': tf.io.FixedLenFeature(shape=(4,), dtype=tf.int64),
    'label': tf.io.VarLenFeature(tf.int64),
    }

    def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码

        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        # shape=feature_dict['shape']
        #print(shape,feature_dict['img'],type(feature_dict['img']),)
        #feature_dict['img'] = tft.sparse_tensor_to_dense_with_shape(feature_dict['img'],shape)    
        #feature_dict['label'] = tft.sparse_tensor_to_dense_with_shape(feature_dict['label'],shape)    
        feature_dict['img']=tf.cast(tf.reshape(tf.sparse.to_dense(feature_dict['img']), shape),dtype=tf.float32)/1.0
        feature_dict['label']=tf.sparse.to_dense(feature_dict['label'])
        print("img",feature_dict['img'])
        if numclass>1:
            feature_dict['label']=tf.keras.utils.to_categorical(feature_dict['label'], num_classes=numclass, dtype='float32')
        # feature_dict['img'] =tf.io.decode_raw(feature_dict['img'], tf.uint16)
        # feature_dict['label'] =tf.io.decode_raw(feature_dict['label'], tf.int32)
        #shape=np.array(shape)
        #print(feature_dict['img'].shape,shape)
        return feature_dict['img'], feature_dict['label']

    dataset = raw_dataset.map(_parse_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def numpy_to_int64_float32(arr):
    if "int" in arr.dtype.__str__() :# change to int64
        return arr.astype(np.int64)
    elif "float" in arr.dtype.__str__():
        return arr.astype(np.float32)
    elif "bool" in arr.dtype.__str__() :# change to int64
        return arr.astype(np.int64)
    
def numpy_value(arr):
    arr=numpy_to_int64_float32(arr).flatten()
    if arr.dtype==np.int64:
        return _int64_feature(arr)
    elif arr.dtype==np.float32:
        return _float_feature(arr)
   
    
def write_tfrecord(imgs, labels,  output_file):
    """Create a training tfrecord file.
        output_file: The file name for the tfrecord file.
    """

    writer = tf.io.TFRecordWriter(output_file)
    for img, label in zip(imgs, labels):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                'img': numpy_value(img), 
                'shape':_int64_feature(list(img.shape)),
                'label': numpy_value(label), 
                }
        ))

        writer.write(example.SerializeToString())
    writer.close()



def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class TFRecordDataset:
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

    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)

    #-----------------------#
    #   write to tfrecord  #
    #-----------------------#
 
    def generate_train_test(self):
        images,segmaps=get_img_label(self.data_path,self.label_path,self.label_suffix,pathonly=False)
        ptr,pva,pte=self.partion
        print(self.partion,"\n total img num : ",len(images))
        X_train, X_test, y_train, y_test=train_test_split(images,segmaps,test_size=1-ptr)
        X_valid, X_test, y_valid, y_test=train_test_split(X_test,y_test,test_size=pte/(pte+pva))
   
        tfrecord_path=self.tfrecord_path
        file_base.create_dir(tfrecord_path)
        trainpath=os.path.join(tfrecord_path,self.train_paths)
        validpath=os.path.join(tfrecord_path,self.validation_paths)
        testpath=os.path.join(tfrecord_path,self.test_paths)

        write_tfrecord(X_train, y_train, trainpath)
        write_tfrecord(X_valid, y_valid, validpath)
        write_tfrecord(X_test, y_test, testpath)
        print("\n".join([trainpath,validpath,testpath]))
    #-----------------------#
    #   load fromtfread  #
    #-----------------------#
    def load_train_test(self,num_class=1):
        tfrecord_path=self.tfrecord_path
        trainpath=os.path.join(tfrecord_path,self.train_paths)
        validpath=os.path.join(tfrecord_path,self.validation_paths)
        testpath=os.path.join(tfrecord_path,self.test_paths)
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            shape=tf.stack([nz, w, w, 1])
        else:
            shape=tf.stack([ w, w, 1])
        trdatasets=read_tfrecord(trainpath,num_class,shape=shape)
        vadatasets=read_tfrecord(validpath,num_class,shape=shape)
        tedatasets=read_tfrecord(testpath,num_class,shape=shape)
        
        trdatasets = trdatasets.batch(self.batch_size)
        vadatasets = vadatasets.batch(self.batch_size)
        tedatasets = tedatasets.batch(self.batch_size)
        
        
        
        return trdatasets,vadatasets,tedatasets

    def test_load_train_test(self):
        tfrecord_path=self.tfrecord_path
        trainpath=os.path.join(tfrecord_path,self.train_paths)
        validpath=os.path.join(tfrecord_path,self.validation_paths)
        testpath=os.path.join(tfrecord_path,self.test_paths)
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            shape=tf.stack([nz, w, w, 1])
        else:
            shape=tf.stack([ w, w, 1])
        datasets=read_tfrecord(trainpath,numclass=1,shape=shape)
        print(datasets)
        #sample_reconstructed = next(dataset.as_numpy_iterator())
        if nz>1:
            for image,label in datasets:
                print(image.shape,image.dtype)
                show_two(image[5,:,:,:],label[5,:,:,:])
        else:
            for image,label in datasets:
                print(image.shape,image.dtype)
                show_two(image,label)

        

if __name__=="__main__":
    tfd=TFRecordDataset()
    #tfd.generate_train_test()
    tfd.test_load_train_test()
    