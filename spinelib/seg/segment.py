
from itertools import cycle

import napari
import numpy as np
import scipy
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage._shared.utils import check_nD, deprecate_kwarg
from skimage.morphology import (binary_dilation, convex_hull_image, dilation,
                                erosion, remove_small_objects)
from skimage.segmentation import expand_labels, watershed
from skimage.segmentation.morphsnakes import _curvop

from ..cflow.meanshift_23D import NormShift, generateShift, sobel_numpy
from ..utils import measure
from ..utils.npixel import array_slice


def div(u):
    Ii=np.gradient(u)
    Norm=np.sqrt(np.sum([I*I for I in Ii]))+1e-8
    Iii=[np.gradient(I,axis=i) for i,I in enumerate(Ii)]
    return sum(Iii)


def cropfromseed(image,ps,boxsize=16):
    """crop image from point seed , and crop size equal to boxsize

    Args:
        image (ndarray): image source
        ps (list,ndarray): a series point is center of crop image, dimension same to image
        boxsize (int,tuple, optional): crop image size, same to image dimension. Defaults to 16.

    Returns:
        _type_: _description_
    """
    if isinstance(boxsize,int):
        boxsize=[boxsize for n in range(image.ndim)]
    boximgs=[]
    # print("image shape" ,image.shape)
    image=np.pad(image,[(s//2,s//2) for s in boxsize])
    for p in ps:
        p=[int(x) for x in p]
        region=array_slice(image,tuple(p),boxsize,center=False,pad=None).copy()
        boximgs.append(region)
        
    return boximgs

#==↓==↓==↓==↓==↓==↓== growth method ==↓==↓==↓==↓==↓==↓#
def foreach_grow(image,num_iter,init_level_set,searchbox,
                     sizeth=np.inf,adth=None,method="geo",
                     smoothing=0,lambda1=1, lambda2=3,
                     oldseg=None,convex=False,
                     usesobel=False,userigid=False):
    """grow area by seed
    Args:
        image (ndarray): z,y,z, 2D/3D
        num_iter (int): max iterration , max spine radius
        init_level_set (label array): label not bool mask
        searchbox (shape): crop area for searching quickly
        sizeth (int, optional): man spine pixel count. Defaults to np.inf.
        adth (mask, optional): neuron mask. Defaults to None.
        method (str, optional): geo or chan. Defaults to "geo".
        smoothing (int, optional): smooth fractor not used. Defaults to 0.
        lambda1 (float, optional): inner factor , lambda1/lambda2 to define border, smaller -> aera larger. Defaults to 1.
        lambda2 (float, optional): outter factor. Defaults to 3.
    Returns:
        ndarray: size same to labels, label grow result
    """
    # print(init_level_set.shape)
    init_level_set=init_level_set.astype(np.int64)
    if adth is not None:
        image=image*adth
    padimage=np.pad(image,[(s//2,s//2) for s in searchbox])
    if oldseg is None or (not oldseg.any()):
        padmask=np.pad(init_level_set,[(s//2,s//2) for s in searchbox])
    else:
        oldseg=oldseg.astype(np.int64)
        init_level_set[oldseg>0]=0
        padmask=np.pad(oldseg+init_level_set,[(s//2,s//2) for s in searchbox])
  
    stn=image.ndim-2
    ndim=image.ndim
    if ndim==3:
        structure=np.ones((1,3,3))
    else:
        structure=np.ones((3,3))
        
    #padadth=np.pad(adth,[(s//2,s//2) for s in searchbox])
    shifts=generateShift(searchbox)
    shifts=NormShift(shifts)
    sobels=sobel_numpy(padimage)
    if usesobel:
        padimage = np.sqrt(sum([grad**2 for grad  in sobels]) / padimage.ndim)
    sobels=NormShift(sobels)
    labs=list(measure.unique_labs(init_level_set))
    
    indxs=np.argwhere(init_level_set>0)
    # if method=="geo":
    #     func=morph_geodesic_contour
    # else:
    #     func=morph_chan_vese
    for ind in indxs:
        indx=tuple(ind)
        lab=init_level_set[indx]
        if lab not in labs:
            continue
        # else:
        #     labs.remove(lab)
        region=array_slice(padimage,tuple(indx),searchbox,center=False,pad=None)
        mask0=array_slice(padmask,tuple(indx),searchbox,center=False,pad=None)
        #adth0=array_slice(padadth,tuple(indx),searchbox,center=False,pad=None)
        mask=mask0==lab
        sobelregion=[array_slice(s,tuple(indx),searchbox,center=False,pad=None) for s in sobels]
        innerdot=sum([shift*sregion for shift,sregion in zip(shifts[stn:],sobelregion[stn:])])
        if userigid:
            rigidmask=innerdot<=-0.1
        else:
            rigidmask=None
        if method=="geo":
            mask2=morph_geodesic_contour(region, num_iter,mask, smoothing=smoothing,balloon=1,rigid=rigidmask,usesobel=usesobel)
        else:
            mask2=morph_chan_vese(region,num_iter,mask,smoothing=0,lambda1=lambda1, lambda2=lambda2,rigid=rigidmask)
        obj = [slice(np.max(int(st),0), int(st+s),1) for st,s in zip(indx,searchbox)]
        # v=napari.Viewer()
        # v.add_image(region)
        # v.add_image(mask)
        # v.add_image(region2)
        # #v.show()
        # plt.figure()
        # plt.show()
       
        #region2[mask>0]=0
        mask2=ndi.binary_fill_holes(mask2,structure=structure).astype(int)
        if convex:
            mask2=fill_hulls(mask2)
        # if (ndim==3):
        #     for m in mask2:
        #mask2 = convex_hull_image(mask2)
        #remove too large object
        objectcount=np.sum(mask2)
        #print(indx,lab,objectcount)
        
        #mask2 : current label grow area
        #mask0 : all label grow area exclude current label, but include current label seed point
        #mask : current seed label
        
        if(objectcount>sizeth or objectcount<4):
            mask2=np.zeros_like(mask2)
            mask0[mask]=0
            
        #merge overlap object
        ovelapmask=mask2*mask0*(~mask)
        overlapcount=np.sum(ovelapmask)
        if (overlapcount): # has overlap
            ar_unique, c = np.unique(ovelapmask, return_counts=True)
            arr1inds = c.argsort()
            sorted_c = c[arr1inds[::-1]]
            sorted_lab = ar_unique[arr1inds[::-1]]
            for clab,c in zip(sorted_c,sorted_lab):
                if clab==0:continue
                if clab==lab:continue
                if overlapcount>0.5*c or c>10 or overlapcount>0.5*objectcount :
                    mask2[mask0==clab]=1        
        mask2=mask2*lab
        mask2[mask2==0]=mask0[mask2==0]
        padmask[tuple(obj)]=mask2
    obj=[slice(s//2,-s//2+1) for s in searchbox]
    return padmask[tuple(obj)] 




def remove_large_objects(ar, max_size=64, connectivity=1,):
    out = ar.copy()
    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")
    too_small = component_sizes > max_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out

def foreach_grow_area(image,num_iter,init_level_set,searchbox,
                     sizeth=np.inf,adth=None,method="geo",
                     smoothing=0,lambda1=1, lambda2=3):
    """grow area by seed
    Args:
        image (ndarray): z,y,z
        num_iter (int): max iterration , max spine radius
        init_level_set (label array): label not bool mask
        searchbox (shape): crop area for searching quickly
        sizeth (int, optional): man spine pixel count. Defaults to np.inf.
        adth (mask, optional): neuron mask. Defaults to None.
        method (str, optional): geo or chan. Defaults to "geo".
        smoothing (int, optional): smooth fractor not used. Defaults to 0.
        lambda1 (float, optional): inner factor , lambda1/lambda2 to define border, smaller -> aera larger. Defaults to 1.
        lambda2 (float, optional): outter factor. Defaults to 3.
    Returns:
        ndarray: size same to labels, label grow result
    """
    padimage=image#np.pad(image,[(s//2,s//2) for s in searchbox])
    padmask=init_level_set#np.pad(init_level_set,[(s//2,s//2) for s in searchbox])
    padadth=adth#np.pad(adth,[(s//2,s//2) for s in searchbox])
    
    # sobels=sobel_numpy(padimage)
    # sobels=NormShift(sobels)
    # labs=list(np.unique(init_level_set))
    # labs.remove(0)
    # ndim=image.ndim
    # indxs=np.argwhere(init_level_set>0)
    mask2=morph_geodesic_contour_area(padimage, num_iter,padmask, smoothing=smoothing,balloon=1,rigid=padadth)
    
    mask2=remove_large_objects(mask2,sizeth)


    return mask2

def morph_geodesic_contour_area(gimage, num_iter,
                                init_level_set='disk', smoothing=1,
                                threshold='auto', balloon=0,
                                rigid=None):
    """Morphological Geodesic Active Contours (MorphGAC).
    """

    image = gimage
    if rigid is None:
        rigid=np.ones_like(image)
    if threshold == 'auto':
        threshold =np.percentile(image, 40)

    structure =np.ones((3,) * image.ndim, dtype=np.int8)#ndimage.generate_binary_structure(image.ndim, 1) 
    dimage = np.gradient(image)
    dimage=NormShift(dimage)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)
    # init_level_set=binary_dilation(init_level_set,structure)*rigid
    u = init_level_set

    for _ in range(num_iter):
        # Balloon
        if balloon > 0:
            new_lset=expand_labels(u)
            aux = new_lset>u
        elif balloon < 0:
            new_lset=erosion(u, structure)
            aux = u>new_lset
        aux=aux*rigid
        # if balloon != 0:
        #     u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        uu = np.zeros_like(dimage[0])
        du = np.gradient((new_lset>0).astype(np.int8))
        #du=NormShift(du)
        
        for el1, el2 in zip(dimage, du):
            uu+= el1 * el2
        mm=(uu*aux) >0
        u[mm]=new_lset[mm]
       

        u=morphology.closing(u)
        u=morphology.opening(u)
        #Smoothing
        for _ in range(smoothing):
            u = _curvop(u)
        
    
    return u
  


def morph_geodesic_contour(gimage, num_iter,
                                          init_level_set='disk', smoothing=1,
                                          threshold='auto', balloon=0,
                                          rigid=None,usesobel=False):
    """Morphological Geodesic Active Contours (MorphGAC).
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
    """

    image = gimage
    if rigid is None:
        rigid=np.ones_like(image)
    if threshold == 'auto':
        threshold =np.percentile(image, 40)

    structure =np.ones((3,) * image.ndim, dtype=np.int8)#ndimage.generate_binary_structure(image.ndim, 1) 
    dimage = np.gradient(image)
    dimage=NormShift(dimage)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)
    init_level_set=binary_dilation(init_level_set,structure)*rigid
    u = np.int8(init_level_set > 0)
    


   
    for _ in range(num_iter):

        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)>u
        elif balloon < 0:
            aux = u>ndi.binary_erosion(u, structure)
        aux=aux*rigid
        # if balloon != 0:
        #     u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        uu = np.zeros_like(dimage[0])
        du = np.gradient(u+aux)
        du=NormShift(du)
        
        for el1, el2 in zip(dimage, du):
            uu+= el1 * el2
        if usesobel:
            mm=(uu*aux) <0
        else:
            mm=(uu*aux) >0.1
        if not (mm.any()): break
        u[mm] = 1
        # u[(uu*aux) < 0] = 0
        
        u=morphology.closing(u)
        u=morphology.opening(u)

        #Smoothing
        for _ in range(smoothing):
            u = _curvop(u)
    if _!=  num_iter:
        return u
    else:
        return np.zeros_like(u)
    
  
    
def morph_chan_vese(image, num_iter, init_level_set='checkerboard',
                            smoothing=1, lambda1=1, lambda2=1,
                            rigid=None):
    """Morphological Active Contours without Edges (MorphACWE)
    """

    labels=init_level_set
    u = np.int8(init_level_set > 0)
    if rigid is None:
        rigid=np.ones_like(image)

  
    # sobels=sobel_filters(image)
    # image = np.sqrt(sum([grad**2 for grad  in sobels]) / image.ndim)

    for _ in range(num_iter):

        # # outside = u <= 0
        # c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-8)
        # # inside = u > 0
        # c1 = (image * u).sum() / float(u.sum() + 1e-8)

        # Image attachment
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        uin=u
        uout=(abs_du>0)>u
        # outside = u <= 0
        c0 = (image * uout).sum() / float(uout.sum() + 1e-8)
        # inside = u > 0
        c1 = (image * uin).sum() / float(uin.sum() + 1e-8)
        # remove indensity large than border candidate point
        # mask=(abs_du>0)>(u>0)
        # uimage=image*u
        # uimage[uimage<1]=65535
        # duimage=image*mask
        # ms=[]
        # for dim in range(image.ndim):
        #     mask[(duimage-10)>np.roll(uimage,1,axis=dim)]=0
        #     mask[(duimage-10)>np.roll(uimage,-1,axis=dim)]=0
        #labels=dilation(labels,cube(3))
        
        aux = abs_du * (lambda1 * (image - c1)**2 - lambda2 * (image - c0)**2)*rigid

        u[aux < 0] = 1
        u[aux > 0] = 0
        
        #labels[u < 1]=0
       # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        

    return u
#==↑==↑==↑==↑==↑==↑== growth method ==↑==↑==↑==↑==↑==↑#

#==↓==↓==↓==↓==↓==↓== convex hull for label roi ==↓==↓==↓==↓==↓==↓#
def fill_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    image :binary
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)
    
    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"
    
    points = np.argwhere(image).astype(np.int16)
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d
    
    mask = np.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:,:,0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask

def fill_hulls(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    iamge : labels int
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)
    
    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"
    
    labs=np.unique(image)
    mask = np.zeros_like(image)
    ndim=image.ndim
    
    if ndim==3:
        for lab in labs:
            if lab<1: continue
            points = np.argwhere(image==lab).astype(np.int16)
            if(len(points)<4):continue
            try:
                hull = scipy.spatial.ConvexHull(points)
                deln = scipy.spatial.Delaunay(points[hull.vertices])
            except:
                mask[image==lab] = lab
                continue

            # Instead of allocating a giant array for all indices in the volume,
            # just iterate over the slices one at a time.
            idx_2d = np.indices(image.shape[1:], np.int16)
            idx_2d = np.moveaxis(idx_2d, 0, -1)

            idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
            idx_3d[:, :, 1:] = idx_2d
            
            
            for z in range(len(image)):
                idx_3d[:,:,0] = z
                s = deln.find_simplex(idx_3d)
                mask[z, (s != -1)] = lab
    else:
        for lab in labs:
            if lab<1: continue
            points = np.argwhere(image==lab).astype(np.int16)
            if(len(points)<4):continue
            try:
                hull = scipy.spatial.ConvexHull(points)
                deln = scipy.spatial.Delaunay(points[hull.vertices])
            except:
                mask[image==lab] = lab
                continue

            idx = np.stack(np.indices(image.shape), axis = -1)
            out_idx = np.nonzero(deln.find_simplex(idx) + 1)
           
            mask[out_idx] = lab

    return mask
#==↑==↑==↑==↑==↑==↑== convex hull for label roi ==↑==↑==↑==↑==↑==↑#



#==↓==↓==↓==↓==↓==↓== resortlable from 1  ==↓==↓==↓==↓==↓==↓#
def resortseg(seg,start=1):
    labellist=list(np.unique(seg))
    labellist.sort()
    arr=seg
    if 0 in labellist:
        labellist.remove(0)
    newl=start-1
    for newl,oldl in enumerate(labellist,start):
        if(newl!=oldl):
            arr[arr==oldl]=newl
    # nstart=len(labellist)
    
    return arr,newl+1
def resortseg_truncate(seg,start=1):
    arr=(seg>=start)*seg
    labellist=list(np.unique(arr))
    if 0 in labellist:
        labellist.remove(0)
    newl=start-1
    for newl,oldl in enumerate(labellist,start):
        if(newl!=oldl):
            arr[arr==oldl]=newl
    # nstart=len(labellist)
    
    return arr,newl+1
#==↑==↑==↑==↑==↑==↑== resortlable from 1 ==↑==↑==↑==↑==↑==↑#


def shrink_label(lables):
    labs=list(measure.unique_labs(lables))
    nlabls=np.zeros_like(lables)
    for lab in labs:
        mask=lables==lab
        mask=erosion(mask)
        nlabls+=mask*lab
    return nlabls

def ndilable(image,offset=1):
    labels,num=ndi.label(image)
    labels[labels>0]+=offset
    return labels,num
    
def remove_small_lable(mask,thsize):
    labs=measure.unique_labs(mask)
    newmask=np.zeros_like(mask)
    for lab in labs:
        mm=mask==lab
        mask2=remove_small_objects(mm,thsize)
        if not mask.any():
            mask2=mm
        newmask[mask2]=lab
   
    return newmask        


def modify_mask(mask,projmask,sizeth=20):
    denmask=mask==1
    spinemask=mask==2
    denmask=remove_small_objects(denmask,min_size=sizeth)
    projmask=remove_small_objects(projmask==1,min_size=sizeth)
    # print(mask.shape[0])
    projmask=np.tile(projmask,(mask.shape[0],1,1))
    spinemask[projmask>0]=0
    mask=spinemask*2+denmask
    return mask

def label_instance_water(img,corner,spinemask,maxspinesize,searchbox=[7,7,7]):
    labels,num=ndilable(corner,1)
    ls = foreach_grow(img, num_iter=4, 
                                init_level_set=labels,
                                searchbox=searchbox,
                                sizeth=maxspinesize,adth=spinemask,
                                method="geo",smoothing=0)
    spine_label=watershed(-img,ls,mask=spinemask)
    return spine_label

def keep_spineimg_bypr(imgs,mask,spinepr=None,th=0.5,cval=None):
    # if not spine pr , th no use ,keep spine img according to mask
    if not cval:
        cval=np.mean(imgs[mask==0])
    imgns=imgs.copy()
    if spinepr is not None:
        imgns[spinepr<th]=cval
    else:
        imgns[mask!=2]=cval
    return imgns

def adth_func_time(imgs,binary_func):
    mim=np.mean(imgs[0:5],axis=0)
    th=binary_func(mim)
    meanbg = np.mean(mim[mim<th])
    adths=[img>th for img in imgs]
    return np.array(adths),th,meanbg


        
#-----------------------#
#   TEST  #
#-----------------------#



def test():
    arr=np.array(
        [[0,0,0,0,0,0,0,0,0],
        [0,2,3,4,0,0,0,0,0],
        [0,4,5,1,0,0,1,0,0],
        [0,1,2,2,0,2,2,5,0],
        [0,0,2,2,0,0,5,0,0],
        [0,0,0,0,0,0,0,0,0],]
    )    
    labels=np.array(
        [[0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,2,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],]
    )    
    mask=foreach_grow(arr,5,labels,[5,5])
   # mask=morph_chan_vese(arr,5,labels)
    print(arr)
    print(labels)
    print(mask)
      

if __name__=="__main__":
    ar=np.arange(9).reshape(3,3)
    ar[0,2]=1
    arr=remove_large_objects(ar,1)
    print(arr)