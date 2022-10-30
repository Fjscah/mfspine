import sys
from operator import index

import numpy as np
import skimage
from pip import main
from sklearn.preprocessing import binarize

sys.path.append(r".")
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_dilation, cube, disk

from .measure import *
from .npixel import *


def path_spine_den(labels):
    """
    Args: straight line(vectors) from spine to nearset dendrite
        labels (int ndarray): den :1 , spine>1, background=0
    return : straight line from spine to nearset dendrite
    """
    fiber=labels!=1
    edt, inds = ndimage.distance_transform_edt(fiber, return_indices=True)
    las,areas,indexs=label_statics(labels)
    vectors=[]
    ndim=labels.ndim
    for index in indexs:
        index=np.round(index,0).astype(np.int)
        index=tuple(index)
        toindex=np.array([ind[index] for ind in inds])
        vector=np.zeros((2,ndim))
        vector[0,...]=index
        vector[1,...]=toindex-np.array(index)
        vectors.append(vector)
    return vectors
    
def constrain_path_spine_den(labels,mask):
    """return paths according to constain distance framsform
    path nlabel=len(paths), for each ele in path, npoint=len(path[0])
    that's [n,np,ndim] list
    for example : 
    [
        [[0,1,2],[0,3,2]],
        [[0,1,2],[0,1,4],[1,2,3]]
    ]

    Args:
        labels (np.ndarray): spine and fiber labels fiber=1, spine =2,3,4,,,,
        mask (np.ndarray):   neuron mask
    """
    labels=labels.copy()
    fiber=labels==1
    ndim=labels.ndim
    if(ndim==3) :footpoint=cube
    elif(ndim==2):footpoint=disk
    diskmatrix=np.zeros_like(labels)
    radius=1
    diff=fiber
    diskmatrix[diff]=radius
    while(diff.any()):
        radius+=1
        dfiber=binary_dilation(fiber,footpoint(3))
        dfiber=dfiber*mask
        diff=dfiber>fiber
        diskmatrix[diff]=radius
        fiber=dfiber
    las,areas,indexs=label_statics(labels)
    paths=[]
    connectlabels={}
    ndim=labels.ndim
    for la,index in zip(las,indexs):
        if(la==1): continue # fiber
        index=np.round(index,0).astype(np.int)
        index=tuple(index)
        path,labels=back_route(index,diskmatrix,labels)
        if len(path)>2:
            paths.append(path)
            #connectlabels[la]=passlabels

    # # merge label
    # lls={}
    # kls={}
    
    # def notindict(vs,lls):
    #     f=True
    #     for v in vs:
    #         if v in lls: 
    #             f=False
    #             break
    #     return f
    # def keysindict(vs,lls):
    #     keys=[]
    #     for v in vs:
    #         if v in lls: 
    #             keys.append(lls[v])
    #     keys=set(keys)
    # for key,vs in connectlabels.items():
    #     #  no any v has been in lls, create new dict
    #     if(notindict(vs,lls)):
    #         for v in vs:
    #             lls[v]=key
    #         kls[key]=set(vs)
    #     else:
    #         keys=keysindict(vs,lls)
    #         for k in keys:
    #             for v in kls[key]:
    #                 lls[v]=key 
    #             kls.pop(k)
    # print("kls",kls)
                
            
    
    
    
    return paths,labels
    
def back_route(index,distancemap,labels):
    size=distancemap.shape
    path=[]
    passlabel=[]
    la=labels[index]
    while(distancemap[index]>1):
        passlabel.append(labels[index])
        points=valid_connect_pixel(index,size) 
        for p in points:
            p=tuple(p)
            if distancemap[p]==distancemap[index]-1:
                path.append(p)
                index=p
                break
    passlabel=set(passlabel)
    if(len(passlabel)>1):
        for lab in passlabel:
            labels[labels==lab]=la
    return path,labels


def nei_label(labels,orientation):
    fiber=labels==1
    ndim=labels.ndim
    if(ndim==3) :footpoint=cube(3)
    elif(ndim==2):footpoint=disk(3)
    labs=list(np.unique(labels))
    labs.remove(0)
    labs.remove(1)
    labdict={}
    labcenters={}
    labos={}
    for lab in labs:
        
        mask=labels==lab
        mask=binary_dilation(mask,footpoint)>mask
        bbs=labels[mask]
        bbs=set(list(bbs))
        if(0 in bbs):
            bbs.remove(0)
        if(1 in bbs):
            bbs.remove(1)
        if(len(bbs)>0):
            labdict[lab]=bbs
    
    def oritation(regionmask,intensity):
        ori=np.nanmean(intensity)
        return ori
    props = regionprops(labels, orientation,extra_properties=(oritation,))
    for prop in props:
        lab=prop.label
        index=prop.centroid
        index=np.round(index,0).astype(np.int)
        index=tuple(index)
        labcenters[lab]=index
        labos[lab]=prop.oritation
    
    for key,vs in labdict.items():
        oinden=labcenters[key]
        oo=labos[key]
        for v in vs:
            to=labos[v]
            tinden=labcenters[v]
            #if np.mod(abs(oo-to),np.pi) < np.pi/4:
            tlab=labels[tinden]
            labels[labels==tlab]=key
            
    return labels
    
def trace_spine_den(labels,fibers,mask):
    """find a line from spine label to fiber, and line not cross to back region

    Args:
        labels (np.ndarray): spine labels ,1,2,3,4,
        fibers (np.ndarray): dendrite fiber, binary
        mask (np.ndarray):   neuron mask
    """
    indxs=np.argwhere(labels>2)
    labellist=[]
    route=np.zeros_like(mask,dtype=np.int8)
    paths=[]
    vetors=[]
    for ind in indxs:
        index=tuple(ind)
        if labels[index] in labellist: continue # has been resolved
        labellist.append(labels[index])
        endp=layerbylayer(index,fibers,mask,route)
        if (endp is None) : # not link to dendrite directly
            pass
        else:
            path,pathlabel=reverseRoute(endp,route,labels)
            pathlabel=set(pathlabel)
            path=np.array(path)
            paths.append(path)
    return paths
def reverseRoute(endp,route,labels):
    """return path from dentrite to spine,and labels pass by (for merge spine)

    Args:
        endp (tuple): dendrite node
        route (np.ndarray): map for get path
        labels (np.ndarray): spine labels

    Returns:
        path,pathlabels: retrun point path ,and label path
    """
    nextpoint=endp
    radius=route[endp]
    newp=None
    path=[endp]
    pathlabel=[]
    while(radius>1):
        radius-=1
        points=valid_connect_pixel(nextpoint,route.shape)
        for neip in points:
            if neip is None:
                continue
            neip=tuple(neip)
            if(route[neip]==radius):
                newp=neip
                path.append(newp)
                pathlabel.append(labels[newp])
                break
        nextp=newp   
    #last radius==1
    route=np.zeros_like(route)
    return path,pathlabel
            
def layerbylayer(start,fibers,mask,route):
    nextpoints=[start]
    route[start]=1
    radius=1
    while(nextpoints):
        newsp=[]
        radius+=1
        for p in nextpoints:
            points=valid_connect_pixel(p,mask.shape)
            for neip in points:
                if neip is None:
                    continue
                neip=tuple(neip)
                if(route[neip]): continue
                if (fibers[neip]): # find fiber node 
                    route[neip]=radius
                    return neip
                if(mask[neip]): # in signal mask
                    newsp.append(neip)
                    route[neip]=radius
        nextpoints=newsp
    return None
    
if __name__ == "__main__":
    m=np.arange(64).reshape((4,4,4))+1
    d=np.eye(4,4,4)>0
    m[d]=1
    print(m)
    vectors=path_spine_den(m)
    print(vectors)
    
    