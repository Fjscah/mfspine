from mahotas import croptobbox
import numpy as np

#=======================#
#   neighbour pixel  #
#=======================#
def IsValid(p,length):
    return p>=0 and p<length
def outOfbound(index,shape):
    if np.min(index)<0:
        return True
    if np.min(shape-np.array(index))<=0:
        return True
    return False
def connect_8(y,x):
    coords=[
        [y-1,x-1],
        [y-1,x],
        [y-1,x+1],
        [y,x-1],
        [y,x+1],
        [y+1,x-1],
        [y+1,x],
        [y+1,x+1],
    ]
    return coords

def connect_26(x,y,z):
    gg,hh,jj=np.mgrid[x-1:x+2,y-1:y+2,z-1:z+2]
    points=[ [x,y,z] for x,y,z in zip(gg.ravel(),hh.ravel(),jj.ravel())]
    points.pop(13) #remove center
    return points
def valid_connect_26(x,y,z,xx,yy,zz):
    points=connect_26(x,y,z)
    for n,(x,y,z) in enumerate(points):
        if IsValid(x,xx) and IsValid(y,yy) and IsValid(z,zz):
            continue
        points[n]=None
    return points

def connect_pixel(point):
    # arbitary dim neighbour pixel
    point=list(point)
    linespace=[np.arange(p-1,p+2) for p in point]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid.ravel() for grid in grids]
    points=[ index for index in zip(*grids)]
    points.pop(3**len(point)//2)
    return points

def valid_connect_pixel(point,size):
    points=connect_pixel(point)
    for n,p in enumerate(points):
        if outOfbound(p,size):
            points[n]=None
    return points



def valid_border_pixel(point,radius,shape):     
    point=list(point)
    linespace=[np.arange(p-radius,p+radius+1) for p in point]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid.ravel() for grid in grids]
    points=[ index for index in zip(*grids)]
    borderpoints=[]
    for p in points:
        if np.max(np.abs(p-point))==radius:
            borderpoints.append(p)
    for n,p in enumerate(borderpoints):
        if outOfbound(p,shape):
            points[n]=None
    return points


def array_slice(arr,start,size,center=False,pad=None):
    if center==True and pad==None:
        obj = [slice(np.max(int(st-s//2),0), int(st+s//2),1) for st,s in zip(start,size)]
        return arr[obj] 
    elif pad==None and center==False:
        obj = [slice(np.max(int(st),0), int(st+s),1) for st,s in zip(start,size)]
        return arr[obj] 
    cropbox=arr[obj]
    if (croptobbox.shape-size).any():
        pass
        #todo
    
    return arr[obj] 

