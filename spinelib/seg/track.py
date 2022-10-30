from ..cflow.blob import peakfilter
from ..cflow.localthreadhold import local_threshold
import numpy as np
from .segment import foreach_grow,foreach_grow_area
from skimage.segmentation import expand_labels
import btrack
from ..imgio import napari_base
from ..utils import npixel
def nms(image,hitpeak,returnmask=True):
    # if clabs is None:
    #     clabs=list(np.unique(hitpeak))
    # if 0 in clabs:
    #     clabs.remove(0)
    indexs=np.argwhere(hitpeak>0)
    indexs=list(indexs)
    values=image[tuple(zip(*indexs))]
    vlabs=hitpeak[tuple(zip(*indexs))]
    zipped_pairs = zip(values,vlabs,indexs)
    #sorted_pairs = sorted(zipped_pairs,lambda x :x[2])
    zipped_pairs = list(zip(vlabs,values,indexs))
    zipped_pairs.sort(key=lambda x:(x[0],-x[1]))
    nmspoints=[]
    viewedlabs=[]
    for lab,value,ind in zipped_pairs:
        if lab in viewedlabs: continue
        else:
            viewedlabs.append(lab)
            nmspoints.append(ind)
    if returnmask:
        mask=np.zeros_like(image)
        mask[tuple(zip(*nmspoints))]=1
        return mask
    else:
        return nmspoints

def keep_lable_seg(segment,labs):
    ls=np.zeros_like(segment)
    for lab in labs:
        ls[segment==lab]=lab
    return ls


def track_by_seg(image,seg,peakmask,adth,
                 num_iter,searchbox,
                 keep_time=3,resvers=None,labcounts={},
                sizeth=np.inf,method="geo",
                smoothing=0,lambda1=1, lambda2=3):
    updatelabs=[]
    if seg is None:
        return np.zeros_like(image),np.zeros_like(image),updatelabs,np.zeros_like(image)
    segment=seg.copy()
    if resvers is not None:
        segment[segment==0]=resvers[segment==0]
        
    hitpeak=(segment*peakmask).astype(np.uint64)
    
    prelab=set(np.unique(segment))
    curlab=set(np.unique(hitpeak))
    
    #intersec
    interseclab=prelab&curlab
    
    #nmsmask=nms(image,hitpeak,returnmask=True)*hitpeak
    #print(np.unique(nmsmask))
    intersecls=foreach_grow(image,num_iter,hitpeak,
                     searchbox=searchbox,sizeth=sizeth,
                     adth=adth,method=method,smoothing=smoothing,
                     lambda1=lambda1,lambda2=lambda2)
        
    #diff # which not find
    diflab=prelab-curlab
    nofinders=keep_lable_seg(segment,diflab)
    for lab in diflab:
        if lab not in labcounts:
            labcounts[lab]=1
        elif labcounts[lab]>keep_time:#expire,erase
            nofinders[nofinders==lab]=0
            labcounts.pop(lab)
        else:
            labcounts[lab]+=1
    # refresh lab ,create update lab
    for lab in interseclab:
        if lab in labcounts:
            updatelabs.append((lab,labcounts[lab]))
            nofinders[nofinders==lab]=0
            labcounts.pop(lab)
    
    return intersecls,nofinders,updatelabs,hitpeak

def label_pre(lss,updatalabs,curls,curframe):
    for lab,count in updatalabs:
        mask=curls==lab
        for n in range(curframe-count,curframe):
            lss[n][mask]=lab

def track_all_series(images,lss,corners,configfile,search_radius=5,tracklength=4):
    ndim=images[0].ndim
    # trans to tracks point
    tracks=[]
    for t,corner in enumerate(corners):
        points=napari_base.get_point(corner) # is list
        tracks+=[[t]+list(p) for p in points]
        # add time
    # save as csv
    print(ndim)
    print("ndim",ndim)
    if ndim==3:
        np.savetxt('test.csv', tracks, delimiter=',',header="t,z,y,x", fmt='%1.1f',comments="")
    elif ndim==2:
        np.savetxt('test.csv', tracks, delimiter=',',header="t,y,x", fmt='%1.1f',comments="")
    # objects = btrack.utils.segmentation_to_objects(
    # segmentation, 
    # properties=('area', ), 
    # )
    #load track
    objects = btrack.dataio.import_CSV('test.csv')
    with btrack.BayesianTracker() as tracker:

        # configure the tracker using a config file
        tracker.configure_from_file(configfile)
        tracker.max_search_radius=search_radius
        # append the objects to be tracked
        tracker.append(objects)

        # set the volume (Z axis volume is set very large for 2D data)
        tracker.volume=((0,1200), (0,1200), (0,1200))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

    #     tracker.export('./test3.hdf5', obj_type='obj_type_1')
        tracks2 = tracker.tracks
        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=ndim)
    newlss=relabel(lss,tracks2,tracklength) 
    return newlss


def _trunc_label(ls,labs):
    mask=np.zeros_like(ls)
    for lab in labs:
        mm=ls==lab
        mask[mm]=lab
    return mask

def _merge_label(ls1,ls2):
    #ls2 add to ls1
    def most_labs(vs,lab):
        ar_unique, c = np.unique(vs, return_counts=True)
        arr1inds = c.argsort()
        sorted_c = c[arr1inds[::-1]]
        sorted_lab = ar_unique[arr1inds[::-1]]
        clab,c=0,0
        for clab,c in zip(sorted_c,sorted_lab):
            if clab==0:continue
            if clab==lab:continue
            return clab,c
        return clab,c
        
    labs=set(np.unique(ls2))
    lab_dict=[]
    for lab in labs:
        mm=ls2==lab
        c2=np.sum(mm)
        vs=ls1[mm]
        clab,c=most_labs(vs,lab)
        if c>0.5*c2:
            lab_dict.append(clab,lab)
            ls1[ls1==clab]=lab
            ls1[mm]=lab
        else:
            ls1[mm]=lab
    return ls1,lab_dict
            
        

def reverse_track_by_seg(imgs,lss,adths,peaks,num_iter,searchbox,keep_time=3,resvers=None,labcounts={},
                sizeth=np.inf,method="geo",
                smoothing=0,lambda1=1, lambda2=3):
    #assume spine not move quickly
    # reverse match
    imgl=len(lss)
    for i1 in range(imgl-1,0,-1):
        ls1=lss[i1]
        ls2=lss[i1-1]
        prelab=set(np.unique(ls1))
        curlab=set(np.unique(ls2))
        diflab=prelab-curlab
        peakmask=peaks[i1-1]
        ls=_trunc_label(ls1,diflab)
        img=imgs[i1-1]
        adth=adths[i1-1]
        hitpeak=(ls*peakmask).astype(np.uint64)
        intersecls=foreach_grow(img,num_iter,hitpeak,
                     searchbox=searchbox,sizeth=sizeth,
                     adth=adth,method=method,smoothing=smoothing,
                     lambda1=lambda1,lambda2=lambda2)
        ls,_=_merge_label(ls2,intersecls)
        lss[i1-1]=ls
        
            
        
def relabel(imgs,tracks,lenth=2):
    ndim=imgs[0].ndim
    size=imgs[0].shape
    newimgs=np.zeros_like(imgs)
    if ndim==3:
        for n,tr in enumerate(tracks,1):
            if len(tr)>=lenth:
                xs=tr.x
                ys=tr.y
                zs=tr.z
                ts=tr.t
                for x,y,z,t in zip (xs,ys,zs,ts):
                    x,y,z,t=int(x),int(y),int(z),int(t)
                    img=imgs[t]
                    oldlab=img[z,y,x]
                    if oldlab!=0:
                        newimgs[t][img==oldlab]=n
                    else:
                        points=npixel.valid_connect_pixel((z,y,x),size)
                        for p in points:
                            if p is not None:
                                p=tuple(p)
                                if img[p]:
                                    oldlab=img[p]
                                    break
                        if oldlab!=0:
                            newimgs[t][img==oldlab]=n
    elif ndim==2:
        for n,tr in enumerate(tracks,1):
            if len(tr)>=lenth:
                xs=tr.x
                ys=tr.y
                ts=tr.t
                for x,y,t in zip (xs,ys,ts):
                    x,y,t=int(x),int(y),int(t)
                    img=imgs[t]
                    oldlab=img[y,x]
                    if oldlab!=0:
                        newimgs[t][img==oldlab]=n
                    else:
                        points=npixel.valid_connect_pixel((y,x),size)
                        for p in points:
                            if p is not None:
                                p=tuple(p)
                                if img[p]:
                                    oldlab=img[p]
                                    break
                        if oldlab!=0:
                            newimgs[t][img==oldlab]=n
    return newimgs
    