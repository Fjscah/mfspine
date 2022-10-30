import numpy as np
from skimage import color, morphology

from skimage.filters.thresholding import (threshold_isodata, threshold_li,
                                          threshold_mean, threshold_minimum,
                                          threshold_multiotsu,
                                          threshold_niblack, threshold_otsu,
                                          threshold_triangle, threshold_yen,threshold_local)
from skimage.morphology import (closing, cube, dilation, opening,
                                remove_small_objects, skeletonize_3d)


def local_threshold_3d(image,filtersize=21,method="gaussian",offset=0):
    
    if(filtersize%2==0): filtersize+=1
    
    oriimage=image.copy()
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    #11 512 512
    mean1=np.mean(image)
    stdv1=np.std(image)

    noiseimg=image.copy()

    noise_mean=np.nanmean(noiseimg[noiseimg<mean1+3*stdv1])
    noise_stdv=np.nanstd(noiseimg[noiseimg<mean1+3*stdv1])

    #sig_mean=np.nanmean(noiseimg[noiseimg>mean1+stdv1])
    #sig_stdv=np.nanstd(noiseimg[noiseimg>mean1+stdv1])
    #print("image mean: ",mean1, " stdv :",stdv1)
    #print("moise mean: ",noise_mean, " stdv :",noise_stdv)

    #%%    ======Threshold========
    ad_th=image.copy()
    #border
    # ad_th_mask=apply_hysteresis_threshold(ad_th, noise_mean+2*noise_stdv,noise_mean+3*noise_stdv)
    ad_th_mask=ad_th>noise_mean+noise_stdv


    #ad_th[ad_th<noise_mean+3*noise_stdv]=0
    #inner smller than mask
    ad_th_inner=ad_th.copy()
    imaget_tophat=morphology.white_tophat(ad_th_inner,morphology.cube(3))
    ad_th_inner=ad_th_inner-imaget_tophat
    ad_th_inner=ad_th_inner>threshold_local(ad_th_inner,(filtersize,filtersize,1),"gaussian",offset=offset)
    # for i in range(image.shape[0]):
    #     imaget=ad_th_inner[i,:,:]
    #     # imaget_tophat=morphology.white_tophat(imaget,morphology.disk(3))
    #     # imaget=imaget-imaget_tophat
    #     ad_th_inner[i,:,:]=imaget>(threshold_local(imaget,filtersize,'gaussian'))

    ad_th_inner[~ad_th_mask]=0

    ad_th_inner=np.array(ad_th_inner,dtype=np.bool_)
    #ad_th_inner=remove_small_objects(ad_th_inner,4)
    return ad_th_inner

def local_threshold_2d(image,winsize=21,offset=0):
    mean1=np.mean(image)
    stdv1=np.std(image)

    noiseimg=image.copy()
    noise_mean=np.nanmean(noiseimg[noiseimg<mean1+3*stdv1])
    noise_stdv=np.nanstd(noiseimg[noiseimg<mean1+3*stdv1])

    sig_mean=np.nanmean(noiseimg[noiseimg>mean1+stdv1])
    sig_stdv=np.nanstd(noiseimg[noiseimg>mean1+stdv1])
    imaget_tophat=morphology.white_tophat(image,morphology.disk(3))
    ad_th_inner=image-imaget_tophat

    # ad_th_mask=apply_hysteresis_threshold(ad_th_inner, noise_mean+noise_stdv,noise_mean+3*noise_stdv)
    ad_th_mask=image>noise_mean+noise_stdv
    
    adaptive=ad_th_inner>threshold_local(ad_th_inner,winsize,'gaussian',offset=offset)
    adaptive[~ad_th_mask]=0
    return adaptive

def local_threshold(image,filtersize=21,mask=None,offset=0):
    if image.ndim==3:
        return local_threshold_3d(image,filtersize,offset=offset)
    else:
        return local_threshold_2d(image,filtersize,offset=offset)
    
def all_threshold_func():
    methods = [threshold_isodata, threshold_li,
            threshold_mean,
            threshold_minimum,
            threshold_multiotsu,
            threshold_niblack, threshold_otsu,
            threshold_triangle, threshold_yen,local_threshold]
    dicts={}
    method_name=[]
    for meth in methods:
        dicts[meth.__name__]=meth
        method_name.append(meth.__name__)
    return dicts,method_name

import numpy as np 

def noisemask(images):
    #---contrust low gradient mask---#
    mean1=np.mean(images)
    stdv1=np.std(images)
    noiseimg=images.copy()
    noise_mean=np.nanmean(noiseimg[noiseimg<mean1+2*stdv1])
    noise_stdv=np.nanstd(noiseimg[noiseimg<mean1+2*stdv1])
    #sig_mean=np.nanmean(noiseimg[noiseimg>mean1+stdv1])
    #sig_stdv=np.nanstd(noiseimg[noiseimg>mean1+stdv1])
    noise_mask=images<noise_mean
    return noise_mask

def remove_label_bysize(label_images,min_size=10,mode="small"):
    """remove small/large label, return small labels
    Args:
        label_images (_type_): _description_
        mode : small/big
    """
    component_sizes = np.bincount(label_images.ravel())
    if (mode=="small"):
        out_range = component_sizes < min_size
    else:
        out_range = component_sizes < min_size
    too_small_mask = out_range[label_images]
    images=label_images.copy()
    images[too_small_mask] = 0
    return images,out_range
