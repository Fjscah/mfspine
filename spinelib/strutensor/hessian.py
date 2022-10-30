from email import header

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   hessian.py
@Time    :   2022/03/26 17:33:40
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

import matplotlib.pyplot as plt

import numpy as np
import plotly.express as px

from ..cflow import *
from skimage import  feature,filters
from itertools import combinations_with_replacement
import numpy.core.numeric as _nx
def gradient_step(f, *varargs, axis=None, edge_order=1,s=1):
    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _nx.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N
    
    
    ss=s*2

    otype = f.dtype
    if otype.type is np.datetime64:
        # the timedelta dtype with the same unit information
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        # view as timedelta to allow addition
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        # result allocation
        out = np.empty_like(f, dtype=otype)

        # spacing for the current axis
        uniform_spacing = np.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(s, -s)
        slice2[axis] = slice(None, -ss)
        slice3[axis] = slice(s, -s)
        slice4[axis] = slice(ss, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2)/(dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = slice(0, s)
            slice2[axis] = slice(s,ss)
            slice3[axis] = slice(0, s)
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = slice(-s,None)
            slice2[axis] = slice(-s,None)
            slice3[axis] = slice(-ss,-s)
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = slice(0, s)
            slice2[axis] = slice(0, s)
            slice3[axis] = slice(s,ss)
            slice4[axis] = slice(ss,2*ss)
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = - dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

            slice1[axis] = slice(-s,None)
            slice2[axis] = slice(-2*ss,-ss)
            slice3[axis] = slice(-ss,-s)
            slice4[axis] = slice(-s,None)
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = - (dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals



#-----------------------#
#   Structure Tensor  #
#-----------------------#
# here put the code
def StructureTensorEle(arr,step=2,sigma=None):
    """return Ix,Iy,[Iz] gauss filter based on sigma, one order

    Args:
        arr (ndarray): image , 2/3 D
        sigma (double): parament for gaussian, =0 no excute gaussian
    retun :  Ix,Iy,[Iz]
    """
    if(sigma is not None):
        arr=ndi.gaussian_filter(arr,sigma,mode='constant', cval=0)
    #Ii=[ndi.sobel(arr,axis=i) for i in range(arr.ndim)]
    Ii=[gradient_step(arr,s=step,axis=i,edge_order=1) for i in range(arr.ndim)]
    return Ii

def get_det_trace_2D(S):
    """return 2D matrix determinant and trace

    Args:
        S (3*ndarray(N*M)): first order gradient matrix A, S=A*AT

    Returns:
        _type_: det and trace
    """
    imageshape=S[0].shape 
    detarr=np.empty_like(S[0])
    tracearr=np.empty_like(S[0])
    harrisarr=np.empty_like(S[0])
    for i in range(imageshape[0]):
        for j in range(imageshape[1]):
            a11=S[0,i,j]
            a12=S[2,i,j]
            a21=S[2,i,j]
            a22=S[1,i,j]
            delv=a11*a22-a12*a21
            trav=a11+a22
            detarr[i,j]=delv
            tracearr[i,j]=trav+0.01 # for divider non zero 
    
    return detarr,tracearr


"""
============================================
Estimate anisotropy in a 3D microscopy image
============================================

In this tutorial, we compute the structure tensor of a 3D image.
For a general introduction to 3D image processing, please refer to
:ref:`sphx_glr_auto_examples_applications_plot_3d_image_processing.py`.
The d2 we use here are imaged from an image of kidney tissue obtained by
confocal fluorescence microscopy (more details at [1]_ under
``kidney-tissue-fluorescence.tif``).

.. [1] https://gitlab.com/scikit-image/d2/#d2

"""
def StructureEvi(image,step=2,sigma=None,mode='constant', cval=0,method="hessian"):
    """cauculate eigenvalue for image 's structure or hessian matrix ,to get three orthoganal lambda

    Args:
        image (ndarray): z,y,z
        step (int, optional): when caculate structure , it need caculate gradient,use spaceing step to caculate,if set hessian , not used. Defaults to 2.
        sigma (None,list of ndim, optional): gaussian filter params, None default set 1. Defaults to None.
        mode (str, optional): _description_. Defaults to 'constant'.
        cval (int, optional): _description_. Defaults to 0.
        method (str, optional): hessian/structure. Defaults to "hessian".

    Returns:
        eigen,stick: stick describe the ballness property
    """

    #print(f'number of dimensions: {image.ndim}')
    #print(f'shape: {image.shape}')
    #print(f'dtype: {image.dtype}')
    if sigma is None: 
        sigma = tuple([1 for i in range(image.ndim)])
    image = feature.corner._prepare_grayscale_input_nD(image)
    if method=="hessian":
        Ii= HessianEle(image, sigma=sigma)
    else:
        Ii=StructureTensorEle(image,step=step,sigma=None)
    
    A_elems = [filters.gaussian(der0 * der1, sigma,mode=mode, cval=cval)
               for der0, der1 in combinations_with_replacement(Ii, 2)]   
     
    #A_elems = feature.structure_tensor(image, sigma=sigma)
    #traceA= A_elems[0]+A_elems[3]+A_elems[5]


    eigen = feature.structure_tensor_eigenvalues(A_elems)
    #print(eigen[0].shape,image.shape,type(image))
    if len(eigen==2):
        stick=eigen[1]*eigen[1]/(eigen[0]-eigen[1]+0.05)
    else:
        stick=eigen[2]*eigen[2]/(eigen[0]-eigen[1]+0.05)/(eigen[1]-eigen[2]+0.05)
    #stick=eigen[2]*eigen[2]/(eigen[0]-eigen[1]+0.05)/(eigen[1]-eigen[2]+0.05)
    #stick2=(eigen[1]-eigen[2])/(eigen[2]+0.001)
    return eigen,stick



#-----------------------#
#   Hession   #
#-----------------------#
def HessianEle(arr,sigma):
    return hessian_matrix(arr, sigma=sigma, order='rc')

def HessianEle_2D(arr,sigma=0):
    """return Ixx,Iyy,Ixy gauss filter based on sigma, two order

    Args:
        arr (ndarray): image , 2 D
        sigma (double): parament for gaussian, =0 no excute gaussian
    retun :  Ixx,Ixy,Iyy
    """
    Ix,Iy=StructureTensorEle(arr,sigma)
    Ixx=np.gradient(Ix,axis=0)
    Iyy=np.gradient(Iy,axis=1)
    Ixy=np.gradient(Iy,axis=0)
    return Ixx,Ixy,Iyy

# here put the code
def HessianEle_3D(arr,sigma=0):
    """return Ixx,Iyy,Ixy gauss filter based on sigma, two order

    Args:
        arr (ndarray): image , 3 D, shape=(nz,nx,ny) 
        sigma (double): parament for gaussian, =0 no excute gaussian
    retun :  Ixx, Ixy, Ixz, Iyy, Iyz, Izz
    """
    Iz,Ix,Iy=StructureTensorEle(arr,sigma)
    
    Izz=np.gradient(Iz,axis=0)
    Ixz=np.gradient(Ix,axis=0)
    Iyz=np.gradient(Iy,axis=0)
    
    #Izx=np.gradient(Ix,axis=1)
    Ixx=np.gradient(Ix,axis=1)
    Iyx=np.gradient(Iy,axis=1)
    
    Iyy=np.gradient(Iy,axis=2)
    
    return Izz,Ixz,Iyz,Ixx,Iyx,Iyy


#-----------------------#
#   ridge  #
#-----------------------#
def enhance_ridges(frame):
    """A ridge detection filter (larger hessian eigenvalue)"""
    shape=frame.shape
    ndim=frame.ndim
    if ndim==3:
        frame=np.max(frame,axis=0)
    blurred = filters.gaussian(frame, 2)
    sigma = 2
    Hxx, Hxy, Hyy = feature.hessian_matrix(blurred, sigma=sigma, mode='nearest', order='xy')
    ridges = feature.hessian_matrix_eigvals([Hxx, Hxy, Hyy])[1]
    ridges=np.abs(ridges)
    thresh = filters.threshold_otsu(ridges)
    thresh_factor = 1.1
    prominent_ridges = ridges > thresh_factor*thresh
    prominent_ridges = morphology.remove_small_objects(prominent_ridges, min_size=128)
    if ndim==3:
        #prominent_ridges=np.expand_dims(prominent_ridges,axis=0)
        prominent_ridges=np.broadcast_to(prominent_ridges[None,...],shape)
    
    
    return prominent_ridges


def create_ridge_mask(frame):
    """"Create a big mask that encompasses all the cells"""
    
    # detect ridges
    ridges = enhance_ridges(frame)
    fig,axs=plt.subplots(1,2)
    # threshold ridge image
    thresh = filters.threshold_otsu(ridges)
    thresh_factor = 1.1
    prominent_ridges = ridges > thresh_factor*thresh
    prominent_ridges = morphology.remove_small_objects(prominent_ridges, min_size=128)
    axs[0].imshow(frame)
    axs[1].imshow(prominent_ridges)
    plt.show()

    # the mask contains the prominent ridges
    mask = morphology.convex_hull_image(prominent_ridges)
    mask = morphology.binary_erosion(mask, disk(10))
    return mask


