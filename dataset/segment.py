import numpy as np
import scipy
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

def resortseg(seg):
    labellist=list(np.unique(seg))
    labellist.sort()
    arr=seg
    for newl,oldl in enumerate(labellist):
        if(newl!=oldl):
            arr[arr==oldl]=newl
    nstart=len(labellist)
    
    return seg,nstart
