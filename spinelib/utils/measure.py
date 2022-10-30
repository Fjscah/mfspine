from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_dilation,cube,disk
import numpy as np
import pandas as pd

def label_statics(label_img):
    # def boundarylabels(regionmask,intensity):
    #     print(intensity.shape)
    #     regionmask=np.bitwise_xor(binary_dilation(regionmask,cube(3)),regionmask)
    #     v=label_img[regionmask]
    #     v=v[v>0]
    #     unique,count=np.unique(v,return_counts=True)
    #     return unique,count

    regions = regionprops(label_img)
    areas=[]
    labels=[]
    indexs=[]
 
    for prop in regions:
        area=prop.area
        label=prop.label
        index = prop.centroid
        
        areas.append(area)
        labels.append(label)
        indexs.append(index)
 
    return labels,areas,indexs

def label_series_statics(imgs,labelss,measure="centroid"):
    """return pandas dataframe for statics data,

    Args:
        labelss (list of ndarray): time series data , 2D/3D
        measure (str, optional): statistic measuement. Defaults to "centroid".
        "area",
        "intensity_mean",
        "image_intensity",
        "centroid",
        #"intensity_mean(th)",
        #"image_intensity(th)",
    """
    lables_dict={}
    # def intensity_mean(regionmask,intenmask):
    #     return np.sum(regionmask)
    print("shape",labelss.shape,labelss.dtype)
    labelss=labelss.reshape(imgs.shape)
    for n,(lables,img) in enumerate(zip(labelss,imgs)):
        regions = regionprops(lables,img)
        for prop in regions:
            label=prop.label
            value=getattr(prop,measure)
            if label in lables_dict:
                lables_dict[label].append(value)
            else:
                lables_dict[label]=[None,]*n+[value]
    df=pd.DataFrame(lables_dict)
    return df
            
def label_statics(label_img):
    # def boundarylabels(regionmask,intensity):
    #     print(intensity.shape)
    #     regionmask=np.bitwise_xor(binary_dilation(regionmask,cube(3)),regionmask)
    #     v=label_img[regionmask]
    #     v=v[v>0]
    #     unique,count=np.unique(v,return_counts=True)
    #     return unique,count

    regions = regionprops(label_img)
    areas=[]
    labels=[]
    indexs=[]
 
    for prop in regions:
        area=prop.area
        label=prop.label
        index = prop.centroid
        
        areas.append(area)
        labels.append(label)
        indexs.append(index)
 
    return labels,areas,indexs
def vector_deg(a,b):
    inner = np.inner(a, b)
    norms = np.linalg.norm(a) *np.linalg.norm(b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))#[0, pi]
    deg = np.rad2deg(rad)
    return deg
def euclidean(x, y):
    x=np.array(x)
    y=np.array(y)
    return np.sqrt(np.sum((x - y)**2))
def unique_labs(labels):
    labs=set(np.unique(labels))
    if 0 in labs:
        labs.remove(0)
    return labs
        

if __name__ == "__main__":
    m=np.arange(64).reshape((4,4,4))+1
    d=np.eye(4,4,4)>0
    m[d]=1
    print(m)
    vectors=label_statics(m)
   
    