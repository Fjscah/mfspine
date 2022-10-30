from matplotlib import lines
from pytest import Instance
from sqlalchemy import false, true
import multiprocessing.dummy
from skimage import filters

import numpy as np
import sys
import scipy.ndimage as ndi
from ..utils import *
# >>> xv, yv = np.meshgrid([0,1,2], [0,1], indexing='ij')
# >>> xv
# array([[0, 0],
#        [1, 1],
#        [2, 2]])
# >>> yv
# array([[0, 1],
#        [0, 1],
#        [0, 1]])
# >>> xv, yv = np.meshgrid([0,1,2], [0,1], indexing='xy')
# >>> xv
# array([[0, 1, 2],
#        [0, 1, 2]])
# >>> yv
# array([[0, 0, 0],
#        [1, 1, 1]])
def generateGaussianKernel(size,sigma=None):
    """generate Gaussian kernel for convolve

    Args:
        size (nd.array): kernel size int , odd beast, [x,y,[z]]
        sigma (nd.array): sigma parament , [sigmax,sigmay,[sigmaz]] ,default 1
        see https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
    """
    #radius =np.round(np.size(size)/2)
    if (sigma is None):
        sigma=[1 for s in size]
    elif isinstance(sigma,float):
        sigma=[sigma for s in size]
    N= np.prod(sigma)*np.sqrt((2*np.pi)**len(size)) #cumulative product
    newsize=size.copy()

    linespace=[np.arange(-(s//2),s//2+1) for s in newsize]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid*grid/sig/sig for grid,sig in zip(grids,sigma) ]
    M=np.exp(-sum(grids))
    G=M/N
    return G

def generateShift(size):
    newsize=size

    linespace=[np.arange(-(s//2),s//2+1) for s in newsize]
    grids = np.meshgrid(*linespace,indexing="ij")
    return grids

def NormShift(grids):
    norms=np.max([np.abs(grid) for grid in grids],axis=0)
    norms[norms<1e-8]=1
    grids=[grid/norms for grid in grids]
    return grids
def weight_shift(shiftkernel,gausskernel,intensitys):
    weights=gausskernel*intensitys
    return weight_shift(shiftkernel,weights)

def weight_shift(shiftkernel,weights):
    return np.sum(shiftkernel*weights)/np.sum(weights)


def outOfbound(index,shape):
    if np.min(index)<0:
        return True
    if np.min(shape-np.array(index))<=0:
        return True
    return False

def meanshift_flow(image,size,sigma):
    """caculating meanshift direction for one iteration base on gaussian filter, 
       then you can get a trace according to continual shift direction line

    Args:
        image (np.array): image input source
        size (list): filter size
        sigma (list): filter sigma params
    Return:
        i
    """
    pad=[(s//2,s//2) for s in size] # len =size.dim
    padding_image=np.pad(image,pad,'constant')
    shifts=[np.zeros_like(image,dtype=float) for s in size] # size.dim*image.shape  pixel diffent axis shift direction and length
    shift_kernels=generateShift(size) # size.dim*size.shape
    gaussian_kernel=generateGaussianKernel(size,sigma) # 1* size.shape
    print("shift kernel number",len(shift_kernels))
    print("shift kernel size",shift_kernels[0].shape)
    print("gaussian kernel size",gaussian_kernel.shape)
    #--- caculate direction ---#
    o=-1
    for index,v in np.ndenumerate(image):
        if(o!=index[0]):
            print("image : ",o)
            o=index[0]
        
        #index_pad=index+np.array(pad)
        intensitys=array_slice(padding_image,index,size)
        #moves=[]
        for i in range(len(size)):
            movei=weight_shift(shift_kernels[i],gaussian_kernel*intensitys)
            #moves.append(movei)
            
            shifts[i][index]=movei
    return shifts

def meanshift_lightpoint(image,size,sigma):
    """caculating meanshift direction for one iteration base on gaussian filter, 
       then you can get a trace according to continual shift direction line

    Args:
        image (np.array): image input source
        size (list): filter size
        sigma (list): filter sigma params
    Return:
        img_Radius : image output source which mark light spot center and radius
    """
    shifts=meanshift_flow(image,size,sigma)
    #--- link direction line---#
    
    img_pointTo=np.zeros_like(image)
    img_pointFrom=np.zeros_like(image)
    # vectors=[]
    for index,v in np.ndenumerate(image):
        moves=[shift[index] for shift in shifts]
       
        if (np.max(np.abs(moves))>0.5):
            # vector=np.zeros((2,2))
            # vector[0,0]=index[0]
            # vector[0,1]=index[1]
            # vector[1,0]=moves[0]
            # vector[1,1]=moves[1]
            # vectors.append(vector)
           
            indexn=np.round(np.array(index)+np.array(moves),0)
            indexn=tuple(np.array(indexn,dtype=int))
            if (outOfbound(indexn,image.shape)):
                
                continue
            else:
                img_pointFrom[index]=1
                img_pointTo[indexn]=1

    #
    img_pointStart=img_pointFrom>img_pointTo
    #img_pointStart=img_pointStart>0
    indxs=np.argwhere(img_pointStart>0)
    result=np.zeros_like(image)
    for ind in indxs:
        index=tuple(ind)
        #print(index)
        while(True):
            f=outOfbound(index,image.shape)
            if (f):
                break
            moves=[shift[index] for shift in shifts]
            if (np.max(np.abs(moves))>0.5):
                index=np.round(np.array(index)+np.array(moves),0)
                index=tuple(np.array(index,dtype=int))
                
            else:
                result[index]+=1
                break
            
 
    #--- filter max radius---#
    
    offset=[s//2 for s in size]
    img_Radius=np.zeros_like(image)
    for index,v in np.ndenumerate(result):
        indexn=tuple(np.array(index)-np.array(offset))
        intensitys=array_slice(result,indexn,size)
        if(intensitys.size==0):
            continue
        if (np.max(intensitys)<=v) :
            img_Radius[index]=v
    #shiftt=sum([shift*shift for shift in shifts])
    img_Radius[img_Radius<2]=0
    labels,nfeature=ndi.label(img_Radius>0)
    print("nfeature:",nfeature)
    # return vectors
    print("done")
    return img_Radius,labels
            

def sobel_numpy_slice(images,mask=None):
    """get gradient flow center from 3D image

    Args:
        images (ndarry): [z,x,y]
        mask (ndarry): which pixel caculate gradient, for reduce computing time

    Returns:
        ndarray: result
    """
    ndim=images.ndim
    if ndim==2 : return sobel_numpy(images,mask)
    
    elif ndim==3:
        nz=images.shape[0]
        gradz=np.zeros_like(images,dtype=float)
        gradx=np.zeros_like(images,dtype=float)
        grady=np.zeros_like(images,dtype=float)
        def calc_one(i):
            gradx[i],grady[i] =np.gradient(images[i])
        pool = multiprocessing.dummy.Pool()
        pool.map(calc_one, range(nz))
    grads=[gradz,gradx,grady]
        
    
    return grads

def sobel_numpy(images,mask=None):
    """get gradient flow center from 3D image

    Args:
        images (ndarry): [[z],x,y]
        mask (ndarry): which pixel caculate gradient, for reduce computing time

    Returns:
        ndarray: result
    """
    #sobels=np.empty(images.shape)
    grads=np.gradient(images)
    #sobels=[filters.sobel(images,axis=i) for i in range(len(images.shape)) ]
    #sobel_mag = np.sqrt(sum([grad**2 for grad  in grads]) / images.ndim)
    
    return grads

def sobel_filters(images,mask=None):
    """get gradient flow center from 3D image

    Args:
        images (ndarry): [[z],x,y]
        mask (ndarry): which pixel caculate gradient, for reduce computing time

    Returns:
        ndarray: result
    """
    
    
    sobels=[filters.sobel(images,axis=i) for i in range(len(images.shape)) ]
 # sobel_mag = np.sqrt(sum([grad**2 for grad  in grads]) / images.ndim)
    
    return sobels

    
    
    



#Test1
# import napari
# import numpy as np
# G=generateGaussianKernel([7,7,7],[3,3,3])
# print("kernel shape : ",G.shape)
# viewer = napari.Viewer()
# new_layer = viewer.add_image(G)
# napari.run()

#tesi2
# a=np.arange(0,9).reshape((3,3))   
# print(a) 
# slicea=array_slice(a,(1,1),(2,2))
# print(slicea)