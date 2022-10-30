import tensorflow as tf
from pint import Measurement
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea,QFileDialog,QMessageBox
from qtpy.QtCore import Qt
import napari
import functools
import numbers
import os
import time
import scipy.ndimage as ndi
from typing import List, Union
from warnings import warn
from .cflow.localthreadhold import all_threshold_func,local_threshold
from .cflow.blob import peakfilter
# from .cflow import *
from .utils.measure import label_series_statics
import napari
import numpy as np
from csbdeep.utils import (
    _raise,
    axes_check_and_normalize,
    axes_dict,
    load_json,
    normalize,
)
from magicgui import magicgui
from magicgui import widgets as mw

from psygnal import Signal

from napari.utils.notifications import show_info

#from . import etsynseg
from .imgio import napari_base
from .seg import segment

from .strutensor.hessian import StructureEvi

import time
import pickle

import numpy as np
import os
from .utils.file_base import split_filename


#-----------------------#
#   paras  #
#-----------------------#
paras=dict(
    steps=2, # caculate stick gradient step lenth 1-4, larger more error , smaller than spine diameter
    sigma=(2,2,2), # z,y,x
    sigma_hessian = 2,
    sigma_supp = 100,
    countth=20, # fiber largest pixels, include z stack, that to say lenth*stackn
    spine_radius=20, # for iteration to growth area
    lambda1=1,
    lambda2=1.5, # lambda2 need large than lambda1
    maxspinesize=800
)

edit_mode = [
    ("None", 0),
    ("Add", 1),
    ("Delete", 2),
]
grow_mode = [
    ("chan_vesc", 0),
    ("geodesic", 1),
]

DEFAULTS = dict(norm_image=True,
                input_scale="None",
                perc_low=1.0,
                perc_high=99.8,
                norm_axes="ZYX",
                prob_thresh=0.5,
                nms_thresh=0.4,
                editmode=edit_mode[0][1],
                growmode=grow_mode[0][1])

Bmethod_func,Bmethod_name=all_threshold_func()

measurement_choices=["all","area",
                     "mean_intensity",
                     "centroid"]#"total_intensity"

Spot_method=["structure","hessian","CNN","SVM"]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#   classify loader  #
#-----------------------#
# svm function
def predict_cls(clff,xs):
    clf,pca,scaler,_=clff
    xs=np.array([x.ravel() for x  in xs])
    xs=scaler.transform(xs)
    xs=pca.transform(xs)
    y_pred = clf.predict(xs)
    return y_pred
def filter_p(points,lables):
    pps=[]
    pps2=[]
    for p,l in zip(points,lables):
        if l>0:
            pps.append(p)
        else:
            pps2.append(p)
    return pps,pps2

def filter_pd(points,image,clfs):
    ndim=image.ndim
    if ndim==3:
        #boxsize=clfs[-1]
        imagemax=np.max(image,axis=0)
        dpsimg=segment.cropfromseed(imagemax,points[...,1:],boxsize=clfs[-1])
    elif ndim==2:
        dpsimg=segment.cropfromseed(image,points,boxsize=clfs[-1])
    y_pred=predict_cls(clfs,dpsimg)
    # print(y_pred)
    points,points2=filter_p(points,y_pred)
    return points,points2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
class CNN(tf.keras.Model):
    def __init__(self,num_class):
        super().__init__()
        self.numclass=num_class
        #self.norm=tf.keras.layers.LayerNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,             # 卷积层神经元（卷积核）数目
            kernel_size=[7, 7],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.numclass)
    @tf.function()
    def call(self, inputs):
        #x=self.norm(inputs)
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        # x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
   
def load_file_model(filepath):
    #lmodel=CNN(2)
    #lmodel.build(input_shape=(None,16,16,1))
    #lmodel.summary() 
    # checkponit_save_path= r"D:\spine\segment\some_analysis_for_spine_img\ipynb\model\temp\variables\variables"
    # if os.path.exists(checkponit_save_path+".index"):
    #     print("------------load the model -------------")
    #lmodel.load_weights(filepath)
    lmodel=tf.keras.models.load_model(filepath, 
    custom_objects=None, compile=True, options=None)
    return lmodel
  
def reshape_data(XX):
    X=[]
    for x in XX:
        # x=filters.sobel(x)
        x=x.reshape(16,16,1)
        x=x.astype(np.float32)
        X.append(x)
    return np.array(X)


def model_predict(points,image,modle,boxsize):
    if image.ndim==3:
        image=np.max(image,axis=0)
        #modle.summary()
        dpsimg=segment.cropfromseed(image,points[...,1:],boxsize=boxsize)
    elif image.ndim==2:
        dpsimg=segment.cropfromseed(image,points,boxsize=boxsize)
    
    # for im in dpsimg:
    #     plt.figure()
    #     plt.imshow(im)
    dpsimg=reshape_data(dpsimg)
    y_p=modle.predict(dpsimg)
    y_pred=y_p.argmax(axis=-1)
    points,points2=filter_p(points,y_pred)
    return points,points2
# lmodel=load_default_model()


### function
# def getdendritemask(img):
#     if img.ndim==3:
#         S, O,B,M = etsynseg.hessian.features3d(img, sigma=paras['sigma_hessian'])
#         B2 = etsynseg.nonmaxsup.nms3d(S, O)
#         nz=img.shape[0]
#     elif img.ndim==2:
#         S, O,B,M = etsynseg.hessian.features2d_H2(img, sigma=paras['sigma_hessian'])
#         B2 = etsynseg.nonmaxsup.nms2d(S, O)
#         nz=1
#     Bsupp, Ssupp = etsynseg.dtvoting.suppress_by_orient(
#     B2, O*B2,
#     sigma=paras['sigma_supp'],
#     dO_threshold=np.pi/6
#     )
#     Ssupp = Ssupp * Bsupp
#     from functools import reduce
#     B_arr = []
#     c_arr = []
#     for c_i, B_i in etsynseg.utils.extract_connected(Bsupp, n_keep=10, connectivity=2,countth=paras['countth']*nz):
#         c_arr.append(c_i)
#         B_arr.append(B_i)
#     B_arr=reduce(np.logical_or,B_arr)
#     B_arr=binary_dilation(B_arr,np.ones((7,) * img.ndim, dtype=np.int8))
#     return B_arr



# widget 1 : training label widget
def plugin_wrapper2():

    #-----------------------#
    #   call back  #
    #-----------------------#
    DEBUG = False

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:

                    print(
                        f'{str(emitter.name).upper()}: {source.name} = {args!r}'
                    )
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    #-----------------------#
    #   GUI  #
    #-----------------------#
    @magicgui(
        image=dict(label="grayscale Image"),
        binaryimage=dict(label="binary Image"),
        spinepoints=dict(label="spine point"),
        denpoints=dict(label="dendrite point"),
        labelmask=dict(label="segment layer"),
        # ambiguspoints=dict(label="ambigus point"),
        preprocess1=dict(widget_type="Label",
                         label="<br><b>1. threshold</b>"),
        axes=dict(widget_type="LineEdit", label="Image Axes", value="zyx"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name,
            value=Bmethod_name[-1],
        ),
        adth_button=dict(widget_type="PushButton", text="binarylize"),
        
        
        preprocess2=dict(widget_type="Label",
                         label="<br><b>2. find seed point</b>"),
        # minrange=dict(widget_type="LineEdit", label="min range", value="5",tooltip=r"/sqrt(3)"),
        # maxrange=dict(widget_type="LineEdit", label="max range", value="9",tooltip=r"/sqrt(3)"),
        # anisotropy_factor=dict(widget_type="LineEdit", label="anisotropy_factor", value="2.0",tooltip=r"float,usually >1,fork from platymatch"),
        spotmethod=dict(
            widget_type="ComboBox",
            label="Spot method",
            choices=Spot_method,
            value=Spot_method[0],
        ),
        model_file=dict(widget_type="FileEdit",visible=False),
        peak_button=dict(widget_type="PushButton", text="find spot"),
        spine_button=dict(widget_type="PushButton", text="to spine"),
        den_button=dict(widget_type="PushButton", text="to den"),
        # ambigus_button=dict(widget_type="PushButton", text="ambiguous"),
        # editrad=dict(
        #     widget_type="RadioButtons",
        #     label="Mode",
        #     orientation="horizontal",
        #     choices=edit_mode,
        #     value=DEFAULTS["editmode"],
        #     tooltip="not work , wait TODO"
        # ),
        
        
        preprocess3=dict(widget_type="Label",
                         label="<br><b>3. grow spine area</b>"),
        growrad=dict(
            widget_type="RadioButtons",
            #visible=False,
            label="grow method",
            choices=grow_mode,
            value=DEFAULTS["growmode"],
            orientation="horizontal",
        ),
        searchbox=dict(widget_type="LineEdit", label="search box", value="11,35,35"),
        smoothing=dict(
            widget_type="SpinBox",
            label="smooth",
            min=0,
            max=3,
            step=1,
            value=0,), 
        lambdav=dict(
            widget_type="FloatSpinBox",
            label="lambda",
            min=0.1,
            max=5,
            step=0.1,
            value=3,), 
        spinesize=dict(
            widget_type="SpinBox",
            label="spine max pixels",
            min=1,
            max=10000,
            step=1,
            value=100,), 
        
        grow_button=dict(widget_type="PushButton", text="run grow"),
        
        
        preprocess4=dict(widget_type="Label",
                         label="<br><b>4. manual modify spine</b>"),
        
        seg_button=dict(widget_type="PushButton", text="reseg"),
        add_button=dict(widget_type="PushButton", text="add label"),
        
        
        
        save_button=dict(widget_type="PushButton", text="save"),
        layout="vertical",
        persist=True,
        call_button=False,  # if True auto add "run" button
    )
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        binaryimage: napari.layers.Image,
        spinepoints:napari.layers.Points,
        denpoints:napari.layers.Points,
        # ambiguspoints:napari.layers.Points,
        labelmask:napari.layers.Labels,
        axes,
        preprocess1,
        bmethod,
        adth_button,
        preprocess2,
        # minrange,
        # maxrange,
        # anisotropy_factor,
        spotmethod,
        model_file,
        peak_button,
        spine_button,
        den_button,
        # ambigus_button,

        preprocess3,
        growrad,
        searchlabel,
        searchbox,
        smoothing,
        lambdav,
        spinesize,
        grow_button,
        preprocess4,
        seg_button,
        add_button,
        save_button,
    ) -> List[napari.types.LayerDataTuple]:
        x = image.data
        axes = axes_check_and_normalize(axes, length=x.ndim)
        lambdav.hide()
        smoothing.hide()
        widgets_valid(plugin.image, valid=False)
        widgets_valid(plugin.binaryimage, valid=False)
        widgets_valid(plugin.labelmask, valid=False)

    #-----------------------#
    #  utils for get value   #
    #-----------------------#
    def get_value_plugin(keystr):
        if keystr=="Binaty_func":
            methodname=plugin.bmethod.value
            func=Bmethod_func[methodname]
            return func
        if keystr=="growmethod":
            return int(plugin.growrad.value)
        if keystr=="searchbox":
            seachstr=plugin.searchbox.value
            lengths=seachstr.split(',')
            lengths=[1-int(l)%2+int(l) for  l in lengths]
            return tuple(lengths)
        if keystr=="lambda":
            return float(plugin.lambdav.value)
        if keystr=="spinesize":
            return int(plugin.spinesize.value)   
        if keystr=="spotrange":
            return int(plugin.minrange.value) ,int(plugin.maxrange.value),float(plugin.anisotropy_factor.value)
        if keystr=="sel_den_point":
            selectpoint=list(plugin.denpoints.value.selected_data)
            if selectpoint:
                return selectpoint,plugin.denpoints.value.data[selectpoint]
            else:
                return selectpoint,None
        if keystr=="sel_spine_point":
            selectpoint=list(plugin.spinepoints.value.selected_data)
            if selectpoint:
                return selectpoint,plugin.spinepoints.value.data[selectpoint]
            else:
                return selectpoint,None
        if keystr=="spotmethod":
            return plugin.spotmethod.value
        if keystr=="modelfile":
            return plugin.model_file.value
        # if keystr=="sel_ambigus_point":
        #     selectpoint=list(plugin.ambiguspoints.value.selected_data)
        #     if selectpoint:
        #         return selectpoint,plugin.ambiguspoints.value.data[selectpoint]
        #     else:
        #         return selectpoint,None
          
            
    # widget_for_methodtype = {
    # StarDist2D: plugin.model2d,
    # StarDist3D: plugin.model3d,
    # CUSTOM_MODEL: plugin.model_folder,
    # }    
    
    #-----------------------#
    #   signal trigger  #
    #-----------------------#
    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet("" if valid else "background-color: lightcoral")
            
   # update = Updater()
    # -----------------------#
    #   event trigger  #
    # -----------------------#
    # set thresholds to optimized values for chosen model
    @change_handler(plugin.adth_button, init=False)
    def _adth_button():
        show_info("plugin image value type"+str(type(plugin.image.value.data)))
        x = plugin.image.value.data
        #x=minimum_filter(x, size=(3,) * x.ndim)
        func=get_value_plugin("Binaty_func")
        #plugin.viewer.value.add_image(x, name="minimum filter")
        if func.__name__=="local_threshold":
            adth=func(x)
        else:
            adth=x>func(x)
        
        plugin.viewer.value.add_image(adth, name="th_"+func.__name__)
        noise_mean=np.mean(x[~adth])
        x=x-noise_mean
        x[x<0]=0
        plugin.image.value.data=x

    @change_handler(plugin.peak_button, init=False)
    def _peak_button():
        img = plugin.image.value.data
        adth= plugin.binaryimage.value.data

        st=time.time()
        ndim=img.ndim
        spotmethod=get_value_plugin("spotmethod")
        corners=peakfilter(img,3,0)*adth
        points=np.array(napari_base.get_point(corners))
        if spotmethod=="hessian" or spotmethod=="structure":
            eigens, stick = StructureEvi(img,method=spotmethod)
            adth2=local_threshold(stick)
            points=np.array(napari_base.get_point(corners*adth2))
            points2=np.array(napari_base.get_point(corners*~adth2))
        elif spotmethod=="CNN":
            mfile=get_value_plugin("modelfile")
            lmodel=load_file_model(mfile)
            points,points2=model_predict(points,img,lmodel,boxsize=(16,16))
        elif spotmethod=="SVM":
            #D:\spine\segment\some_analysis_for_spine_img\ipynb\clf.pickle
            fclf=r"D:\code\myspine\models\svm_model\clf.pickle"
            mfile=get_value_plugin("modelfile")
            with open(mfile, 'rb') as f:
                clff = pickle.load(f)
            points,points2=filter_pd(points,img,clff)
        # corners=napari_base.get_mask_from_point(points,img.shape)

        et=time.time()
        show_info("find peak number : "+str(len(points))+"\nrun time:"+str(et-st)+"s\n")
        plugin.viewer.value.add_points(points,size=4,opacity=0.5,name="spine point",face_color="blue")
        plugin.viewer.value.add_points(points2,size=4,opacity=0.5,name="dendrite point",face_color="red")
        #plugin.viewer.value.add_points([np.zeros(ndim)],size=4,opacity=0.5,name="ambigus point",face_color="yellow")
        #plugin.viewer.value.add_image(adth*adth2,name="distance")
    
    @change_handler(plugin.grow_button, init=False)
    def _grow_button():
        img = plugin.image.value.data
        adth= plugin.binaryimage.value.data
        points=plugin.spinepoints.value.data
        st=time.time()
        seedmask=napari_base.get_mask_from_point(points,img.shape)
        seedmask,_=ndi.label(seedmask)
        growmethod=get_value_plugin("growmethod")
        if growmethod==1:
            growmethod="geo"
        elif growmethod==0:
            growmethod="chan"
        
        searchbox=get_value_plugin("searchbox")
        if len(searchbox)!=img.ndim:
            messg=QMessageBox(None)
            messg.setText("search box must be "+str(img.ndim)+" dimension!")
            messg.exec_()
            return
        num_iter=np.max(searchbox)
        lambdav=get_value_plugin("lambda")
        spinesize=get_value_plugin("spinesize")
        ls=segment.foreach_grow(img,num_iter,seedmask,searchbox,
                     sizeth=spinesize,adth=adth,method=growmethod,
                     smoothing=0,lambda1=1, lambda2=lambdav)
        seedmask[ls<1]=0
        points=napari_base.get_point(seedmask)
        plugin.spinepoints.value.data=points
        et=time.time()
        show_info("run grow method :"+growmethod+"\nrun time:"+str(et-st)+"s\n")
        plugin.viewer.value.add_labels(ls, name="segment")
    @change_handler(plugin.seg_button, init=False)
    def _seg_button():
        img = plugin.image.value.data
        adth= plugin.binaryimage.value.data
        preseg=plugin.labelmask.value.data
        preseg,startlab=segment.resortseg(preseg)
        points=plugin.spinepoints.value.data
        if not points.any(): return
        seedmask=napari_base.get_mask_from_point(points,img.shape)
        seedmask=seedmask>preseg
        seedmask,_=ndi.label(seedmask)
        seedmask[seedmask>0]+=startlab
        growmethod=get_value_plugin("growmethod")
        if growmethod==1:
            growmethod="geo"
        elif growmethod==0:
            growmethod="chan"
        
        show_info("run grow method :"+growmethod)
        searchbox=get_value_plugin("searchbox")
        num_iter=np.max(searchbox)
        lambdav=get_value_plugin("lambda")
        spinesize=get_value_plugin("spinesize")
        ls=segment.foreach_grow(img,num_iter,seedmask,searchbox,
                     sizeth=spinesize,adth=adth,method=growmethod,
                     smoothing=0,lambda1=1, lambda2=lambdav,oldseg=preseg)
        plugin.labelmask.value.data=ls*adth
        seedmask[ls<1]=0
        points=napari_base.get_point(seedmask)
        #plugin.points.value.data=points
        #plugin.viewer.value.add_labels(ls, name="segment")
    @change_handler(plugin.add_button, init=False)
    def _add_button():
        preseg=plugin.labelmask.value.data
        preseg,startlab=segment.resortseg(preseg)
        plugin.labelmask.value.selected_label=startlab
    @change_handler(plugin.spine_button, init=False)
    def _spine_button():# den to spine
        indx,points=get_value_plugin("sel_den_point")
        print(indx)
        if indx:
            plugin.denpoints.value.data=np.delete(plugin.denpoints.value.data,indx,axis=0)
            plugin.spinepoints.value.data=np.concatenate((plugin.spinepoints.value.data,points),axis=0)
    @change_handler(plugin.den_button, init=False)
    def _den_button():# den to spine
        indx,points=get_value_plugin("sel_spine_point")
        print(indx)
        
        if indx:
            plugin.spinepoints.value.data=np.delete(plugin.spinepoints.value.data,indx,axis=0)
            plugin.denpoints.value.data=np.concatenate((plugin.denpoints.value.data,points),axis=0)    
    # @change_handler(plugin.ambigus_button, init=False)
    # def _ambigus_button():# den to spine
    #     indx,points=get_value_plugin("sel_spine_point")
    #     print(indx)
    #     if indx:
    #         plugin.spinepoints.value.data=np.delete(plugin.spinepoints.value.data,indx,axis=0)
    #         plugin.ambiguspoints.value.data=np.concatenate((plugin.ambiguspoints.value.data,points),axis=0)
    #     indx,points=get_value_plugin("sel_den_point")
    #     print(indx)
    #     if indx:
    #         plugin.denpoints.value.data=np.delete(plugin.denpoints.value.data,indx,axis=0)
    #         plugin.ambiguspoints.value.data=np.concatenate((plugin.ambiguspoints.value.data,points),axis=0)    
            
    @change_handler(plugin.save_button, init=False)
    def _save_button():# den to spine
        filename, _ = QFileDialog.getSaveFileName(None, "Save  as...", ".", "*.npy")
        if filename:
            image = plugin.image.value.data
            ndim=image.ndim
            dps=plugin.denpoints.value.data[:,ndim-2:]
            #print(dps)
            sps=plugin.spinepoints.value.data[:,ndim-2:]
            # ambigus=plugin.ambiguspoints.value.data[:,ndim-2:]
            
            if ndim==3:
                image=np.max(image,axis=0)
            dpsimg=segment.cropfromseed(image,dps,boxsize=(16,16))
            spsimg=segment.cropfromseed(image,sps,boxsize=(16,16))
            # ambimg=segment.cropfromseed(image,ambigus,boxsize=(16,16))
            images=dpsimg+spsimg
            #t=np.array(images)
            #print(np.unique(t))
            #print(t.shape,dpsimg[0].shape)
            #print(t.shape[2])
            print("point number : ",len(images))
            lables=[0,]*len(dpsimg)+[1,]*len(spsimg)
            np.save(filename,[images,lables,[16,16]])
            
            show_info("save file : "+filename)
    @change_handler(plugin.spotmethod, init=False)
    def _spot_method(model_name_star: str): 
        spotm=get_value_plugin("spotmethod") 
        #show_info("KKKKKKKKK") 
        if spotm in ["CNN","SVM"]:
            plugin.model_file.show() 
        else:
            plugin.model_file.hide()
    
    return plugin


def th_img(imgs,all_lls,binary_func):
    global_th=binary_func(imgs[0])
    lss=np.zeros_like(all_lls)
    for i in range(len(imgs)):
        img=imgs[i]
        adth=img>global_th
        #corner=peakfilter(img,3,0,use_gaussian=True)*adth
        ls=all_lls[i].copy()
        #ls[~adth]=0
        lss[i]=ls
    return lss


#widget 2 : time series tracking widget
def plugin_wrapper3():

    #-----------------------#
    #   call back  #
    #-----------------------#
    DEBUG = False

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:

                    print(
                        f'{str(emitter.name).upper()}: {source.name} = {args!r}'
                    )
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    #-----------------------#
    #   GUI  #
    #-----------------------#
    @magicgui(
        image=dict(label="grayscale Image"),
        labelmask=dict(label="segment layer"),
        axes=dict(widget_type="LineEdit", label="Image Axes", value="tzyx"),
        preprocess1=dict(widget_type="Label",label="<br><b>Measurement</b>"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name[:-1],
            value=Bmethod_name[0],
        ),
        area_button=dict(widget_type="PushButton", text="get area"),
        mmethod=dict(
            widget_type="ComboBox",
            label="Measurement",
            choices=measurement_choices,
            value=measurement_choices[0],
        ),
        save_button=dict(widget_type="PushButton", text="save measurement"),
        layout="vertical",
        persist=True,
        call_button=False,  # if True auto add "run" button
    )
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        labelmask:napari.layers.Labels,
        axes,
        preprocess1,
        bmethod,
        area_button,
        mmethod,
        save_button
      
    ) -> List[napari.types.LayerDataTuple]:
        x = image.data
        axes = axes_check_and_normalize(axes, length=x.ndim)


    #-----------------------#
    #  utils for get value   #
    #-----------------------#
    def get_value_plugin(keystr):
        if keystr=="measure":
            return plugin.mmethod.value
        if keystr=="Binaty_func":
            methodname=plugin.bmethod.value
            func=Bmethod_func[methodname]
            return func
    @change_handler(plugin.area_button, init=False)
    def _area_button():
        imgs=plugin.image.value.data
        labless=plugin.labelmask.value.data
        #mmethod=get_value_plugin("measure")
        #show_info("measurement : "+mmethod)
        #filename, _ = QFileDialog.getSaveFileName(None, "Save  as...", ".", "*.csv")
        #dirpath,shortname,suffix=split_filename(filename)
        # change imgs
        func=get_value_plugin("Binaty_func")
        labless=th_img(imgs,labless,func)
        #print(labless.shape,labless.dtype)
        plugin.viewer.value.add_labels(labless, name="area segment")
    @change_handler(plugin.save_button, init=False)
    def _save_button():
        imgs=plugin.image.value.data
        labless=plugin.labelmask.value.data
        mmethod=get_value_plugin("measure")
        show_info("measurement : "+mmethod)
        filename, _ = QFileDialog.getSaveFileName(None, "Save  as...", ".", "*.csv")
        dirpath,shortname,suffix=split_filename(filename)
        if mmethod=="all":
            for mm in measurement_choices[1:]:
                # if "area" in mm:
                #     func=get_value_plugin("Binaty_func")
                #     lss=th_img(imgs,labless,func)
                #     df=label_series_statics(imgs,lss,mmethod)
                #     outfile=os.path.join(dirpath,shortname+"_"+mmethod+suffix)
                #     df.to_csv(outfile)
                #     show_info(outfile)
                #     # plugin.viewer.value.add_labels(lss, name="area segment")
                    
                # else:
                df=label_series_statics(imgs,labless,mm)
                outfile=os.path.join(dirpath,shortname+"_"+mm+suffix)
                df.to_csv(outfile)
                show_info(outfile)             
        else :
            # if "area" in mmethod:
            #     # change imgs
            #     func=get_value_plugin("Binaty_func")
            #     labless=th_img(imgs,labless,func)
            #     print(labless.shape,labless.dtype)
            #     # plugin.viewer.value.add_labels(labless, name="area segment")
            df=label_series_statics(imgs,labless,mmethod)
            outfile=os.path.join(dirpath,shortname+"_"+mmethod+suffix)
            df.to_csv(outfile)
            show_info(outfile)
            
 
    return plugin

