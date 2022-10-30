#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   napari_base.py
@Time    :   2022/04/13 21:22:06
@Author  :   Fjscah 
@Version :   1.0
@License :   GNU, LNP2 group
@Desc    :   some transform between napari layer and numpy.array
'''

# here put the import lib


import napari
import numpy as np 
import multiprocessing
import multiprocessing.dummy



def NormShift(grids):
    norms=np.max([np.abs(grid) for grid in grids],axis=0)
    norms[norms<1e-8]=1
    grids=[grid/norms for grid in grids]
    return grids

def orientation2vectors(images,mode="2D",mask=None):
    """transfrom images'orientation to napari' vector for visualzation

    Args:
        images (ndarray): orientation
        mode (str, optional): 2D, 3D . if 2D and images is 3D , will process as 2D slice. if image is 2D, force as 2D. Defaults to "2D".
        mask : only mask area will generate vectors
    """
    
    # vectors format n*2*ndim , n is vector number, from point [n,0,:] to [n,1,:] 
    vectors=[]
    if (mask is None):
        mask=np.ones_like(images)
    indxs=np.argwhere(mask>0)
    ndim=images.ndim
    if(mode=="2D"):
        if(ndim >2): # default z,x,y
            for ind in indxs:
                index=tuple(ind)
                oritation=images[index]
                dx=np.cos(oritation)
                dy=np.sin(oritation)
                
                vector=np.zeros((2,ndim))
                vector[0,...]=index
                vector[1,...]=(0,dy,dx)
                vectors.append(vector)
        else: # xy
            for ind in indxs:
                index=tuple(ind)
                oritation=images[index]
                dx=np.cos(oritation)
                dy=np.sin(oritation)
                
                vector=np.zeros((2,ndim))
                vector[0,...]=index
                vector[1,...]=(dy,dx)
                vectors.append(vector)
    elif(mode=="3D"):
        # Todo
        pass
    return np.array(vectors)

def grad2vectors(grads,mask=None):
    """transfrom images'gradient to napari' vector for visualzation
    """
    grads=NormShift(grads)
    
    # vectors format n*2*ndim , n is vector number, from point [n,0,:] to [n,1,:] 
    vectors=[]
    points=[]
    if (mask is None):
        mask=np.ones_like(grads[0])
    indxs=np.argwhere(mask>0)
    ndim=grads[0].ndim
    indxs=np.argwhere(mask>0)

    for ind in indxs:
        index=tuple(ind)
        vec=np.array([grad[index] for grad in grads])
        
        vector=np.zeros((2,ndim))
        vector[0,...]=index
        vector[1,...]=vec
        vectors.append(vector)
        points.append(index)

    return np.array(vectors),points

def get_point(mask):
    indexs=np.argwhere(mask>0)
    points=list(indexs)
    # for ind in indexs:
    #     index=tuple(ind)
    #     points.append(index)
    return points

def get_mask_from_point(points,imgsize):
    mask=np.zeros(imgsize)
    points=np.around(points,0)
    points=np.array(points,dtype=np.uint64)
    indexs=list(points)
    mask[tuple(zip(*indexs))]=1

    return mask