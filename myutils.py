import numpy as np
import nibabel as nib
import os
import struct
from statsmodels.api import OLS,add_constant
from os.path import join as opj
import math
#import ipdb # Had to downgrade to 0.8.1 due to bug
from IPython.core.debugger import Tracer
from mayavi import mlab
import matplotlib.pyplot as plt

# Sklearn stuff
from sklearn.decomposition import TruncatedSVD
import sklearn.metrics.cluster as metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster.hierarchical import _hc_cut # Internal function to cut ward tree, helps speed up things a lot
from sklearn.utils import resample
from sklearn.model_selection import KFold,StratifiedKFold

def balance_svd_components(trg,src,return_balanced=False):
    nt=np.shape(trg)[1]
    ns=src.shape[1]
    trg_ind=np.arange(0,nt)
    src_ind=np.zeros(ns,dtype=int)
    flip_ind=np.ones(ns,dtype=float)
    for nk in np.arange(0,nt):
        cc=[np.dot(src[:,nk].T,trg[:,ct]) for ct in trg_ind]
        ind=np.argmax(np.abs(cc))
        src_ind[nk]=trg_ind[ind]
        trg_ind=np.delete(trg_ind,ind)
        flip_ind[nk]=np.sign(cc[ind])

    # Leave left over components in the same order
    if nt<ns:
        src_ind[nt:ns]=np.arange(nt,ns)

    if return_balanced:
        return src_ind,flip_ind,np.dot(src,np.diag(flip_ind))[:,src_ind]
    else:
        return src_ind,flip_ind

def assert_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def vec_angle(v1,v2):
    return np.arccos(np.dot(v1/vlen(v1), v2/vlen(v2)))

def norm_arclen(v,n):
    return vlen(v)**2*np.arcsin(np.dot(n,v)/vlen(v))/np.dot(n,v)

def norm_range(x,percentile=None):
    if x.ndim!=1: # Until it is extended to matrices
            raise ValueError('Input data needs to be a vector')
    if percentile is not None: # Trime data above and below percentiles
        if len(percentile)!=2 and percentile.dtype!=float:
            raise ValueError('Percentile need to be specified by an array containing exactly two floats')
        lp=np.percentile(x,percentile[0])
        up=np.percentile(x,percentile[1])
        x[x<lp]=lp
        x[x>up]=up
    return (x - x.min()) / (x.max() - x.min())

def rotate3d(v,ang,axis=0,compute=True,matrix=False):
    # Simple 3d rotation, one axis at a time, to avoid confusion.
    # Could be expanded eventually, but not necessary at the time

    if axis==0: # x
        R=np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])
    elif axis==1: # y
        R=np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])
    elif axis==2: #z
        R=np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])

    if compute and matrix:
        return np.dot(R,v),R
    if not compute and matrix:
        return R
    if compute and not matrix:
        return np.dot(R,v)

def vlen(x,axis=-1):
    return np.sum(np.abs(x)**2,axis=axis)**(1./2)

def normv(v):
    return v/vlen(v)

def normm(data,axis=-1):
    return np.apply_along_axis(normv,axis,data)

def ward_clustering(data,adjacency,mode='split',K_range=None,save_out=None,mask=None,N_iter=1000,svd=False,K_svd=None,group=None,verbose=False):
    # Perform Ward clustering on BPnd data

    if K_range is None:
        raise ValueError('Range of cluster number not specified')
    if svd:
        svd=TruncatedSVD(n_components=K_svd,algorithm='arpack')

    if mode=='whole':
        if verbose:
            print('Performing ward clustering of the whole dataset')

        # Apply func to the data
        if data.dtype==object:
            data=np.column_stack(data)
        if svd:
            svd.fit(data.T)
            fdata=svd.components_.T
        else:
            fdata=data

        ward_labels=np.zeros([fdata.shape[0],len(K_range)])
        for k,nk in zip(K_range,np.arange(0,len(K_range))):
            mdl=AgglomerativeClustering(n_clusters=k, connectivity=adjacency,linkage='ward')
            mdl.fit(fdata)
            ward_labels[:,nk]=mdl.labels_+1

        if save_out is not None:
            if mask is None:
                raise ValueError('Specifiy a surface mask to save data')
            fs_save_surf_data(ward_labels,save_out,mask=mask,verbose=verbose)

    # Evaluate clustering stability for a range of K
    elif mode=='split':
        if verbose:
            print('Performing split-half evaluation of clustering')

        ars=np.empty([N_iter,len(K_range)])
        ami=np.empty([N_iter,len(K_range)])
        mdl1=AgglomerativeClustering(compute_full_tree=True, connectivity=adjacency,linkage='ward')
        mdl2=AgglomerativeClustering(compute_full_tree=True, connectivity=adjacency,linkage='ward')
        if data.dtype==object:
            N=len(data)
        else:
            N=data.shape[1]

        for ni in np.arange(0,N_iter):
            # Compute splits
            if group is None:
                kf=KFold(n_splits=2,shuffle=True,random_state=None) # Split-half model
                split1,split2=kf.split(np.arange(0,N))
            else:
                kf=StratifiedKFold(n_splits=2,shuffle=True,random_state=None) # Split-half model
                split1,split2=kf.split(np.arange(0,N,group))

            # Apply SVD
            if data.dtype==object:
                data1=np.column_stack(data[split1[0]])
                data2=np.column_stack(data[split2[0]])
            else:
                data1=data[:,split1[0]]
                data2=data[:,split2[0]]
            if svd:

                svd.fit(data1.T)
                fdata1=svd.components_.T
                svd.fit(data2.T)
                fdata2=svd.components_.T
            else:
                fdata1=data1
                fdata2=data2

            mdl1.fit(fdata1)
            mdl2.fit(fdata2)

            for nk in np.arange(0,len(K_range)):
                # Cut trees
                # NOTE: labels start at 0, so add one to distinguish from medial wall
                labels1=_hc_cut(nk,mdl1.children_,mdl1.n_leaves_)+1
                labels2=_hc_cut(nk,mdl2.children_,mdl2.n_leaves_)+1

                # Compute metrics
                ars[ni,nk]=metrics.adjusted_rand_score(labels1, labels2)
                ami[ni,nk]=metrics.adjusted_mutual_info_score(labels1, labels2)

        # Plot metrics
        plt.figure(figsize=(8,2))
        plt.subplot(1,2,1)
        plt.plot(K_range,ars.mean(axis=0))
        plt.title('Adjusted Rand Index')
        plt.subplot(1,2,2)
        plt.plot(K_range,ami.mean(axis=0))
        plt.title('Adjusted Mutual Information')
        plt.show()

# FreeSurfer stuff

def fs_read_surf(fname,verbose=False):

    # This function is derived from freesurfer_read_surf.m
    # Note: files are encoded in big-endian
    # Note: Nibabel also has this function, but it should be equivalent

    with open(fname,mode='rb') as f:
        # Verify that file is a triangle file
        magic=f.read(3)
        if magic!=b'\xff\xff\xfe':
            f.close()
            raise ValueError('Invalid magic numnber, file probably does not containg mesh triangle information.')

        # Skip data and info line
        f.readline()
        f.readline()

        # Read the number of faces and vertices
        nv=struct.unpack('>i',f.read(4))[0] # int32
        nf=struct.unpack('>i',f.read(4))[0] # int32
        if verbose:
            print('Reading ' + str(nf) + ' faces and ' + str(nv) + ' vertices.')

        # Read data and reshape
        position=np.reshape(struct.unpack('>' + str(3*nv) + 'f',f.read(nv*12)),[nv,3],order='C') # 3 * float32
        faces=np.reshape(struct.unpack('>' + str(3*nf) + 'i',f.read(nf*12)),[nf,3],order='C') # 3 * int32

    f.close()
    return faces,position

def fs_read_label(fname):
    # Note: Nibabel also has this function, but it should be equivalent
    with open(fname) as f:
        for i,line in enumerate(f): # i starts at 0
            if i==1: # Line 2 contains the number of vertices in the label
                nv=np.zeros(int(line[:-1]),dtype=int)
            elif i>1: # Two first lines are header
                nv[i-3]=int(line.split()[0])
    f.close()
    return nv

def fs_surf_neighborhood(fname,out_type='matrix',mask=None,save_out=None,verbose=False):

    if out_type not in {'matrix','list'}:
        raise ValueError('Invalid out_type ' + out_type)

    # Build neighborhood matrix where neigh(i,j)=True is i and j are neighbors, False otherwise.
    if verbose:
        print('Reading ' + fname)
    faces,_=fs_read_surf(fname)
    if mask is not None:
        vertices,_,_=fs_load_mask(mask)
    else:
        vertices=np.unique(faces) # Unique vertices ID
    vertices=np.sort(vertices)

    # Preallocate memory
    if out_type=='list':
        neigh=np.ndarray(len(vertices),dtype=object)
    else:
        neigh=np.zeros([len(vertices),len(vertices)],dtype=int)

    # Identify neighbors
    if verbose:
        print('Creating neighborhood structure')
    for nv,ni in zip(vertices,np.arange(0,len(vertices))):
        if verbose and ni % 5000==0 and ni != 0:
            print(str(ni) + '/' + str(len(vertices)))
        neigh_ids=np.unique(faces[np.sum(faces==nv,axis=1)>=1,:])
        # Here we are only comparing to vertices in case a mask was provided
        if out_type=='matrix':
            neigh[np.array([x in neigh_ids and x!=nv for x in vertices],dtype=bool),ni]=1
        else:
            neigh[ni]=neigh_ids[np.array([x in vertices and x!=nv for x in neigh_ids],dtype=bool)]

    if save_out is not None:
        if verbose:
            print('Saving neighborhood structure to ' + save_out + '.' + out_type)
        np.savez(save_out + '.' + out_type,neigh,vertices)

    return neigh,vertices

def fs_load_surf_neighborhood(fname):
    if not fname.endswith('.npz'):
        fname=fname+'.npz'
    out=np.load(fname)
    return out['arr_0'],out['arr_1']

def fs_create_cortex_mask(subjects_dir,targ,hemi,validate=True,verbose=False,save_out=None):

    if verbose:
        print('Reading labels')
    cortex=np.sort(fs_read_label(opj(subjects_dir,targ,'label',hemi+'.cortex.label')))
    medial=np.sort(fs_read_label(opj(subjects_dir,targ,'label',hemi+'.Medial_wall.label')))
    N=len(cortex)+len(medial)

    # Make sure each cortex vertice is neighbor of at least another cortex vertice
    # This sanity check is necessary since this vertices of the medial wall have previously
    # been labeled as cortex for fsaverage, left hemisphere (e.g. fsaverage/surf/lh.pial)
    if validate:
        if verbose:
            print('Validating cortex mask, this may take a while')
        faces,_=fs_read_surf(opj(subjects_dir,targ,'surf',hemi+'.pial'),verbose=verbose)
        invalid=np.ndarray(0,dtype=int)
        for nc,ni in zip(cortex,np.arange(0,len(cortex))):
            if verbose:
                if ni % 10000==0 and ni != 0:
                    print(str(ni) + '/' + str(len(cortex)))
            if np.sum([nv in cortex for nv in np.unique(faces[np.sum(faces==nc,axis=1)>=1,:])])<3: # Require at least two neighbors in cortex
                if verbose:
                    print('Detected invalid vertex ' + str(nc))
                np.append(invalid,nc)
        cortex=cortex[np.array([nc not in invalid for nc in cortex],dtype=bool)]
        medial=np.append(medial,invalid)

    if save_out is not None:
        if verbose:
            print('Saving masks to ' + save_out)
        np.savez(save_out,cortex,medial,N)

    return cortex,medial,N

def fs_load_mask(fname):
    if not fname.endswith('.npz'):
        fname=fname+'.npz'
    out=np.load(fname)
    return out['arr_0'],out['arr_1'],out['arr_2']

def fs_load_surf_data(fname,mask=None,output_mask=False):
    img=nib.load(fname)
    img=np.squeeze(img.get_data())
    if mask is not None:
        ma,_,_=fs_load_mask(mask)
        if img.ndim>1 and img.shape[1]>1:
            img=img[ma,:]
        else:
            img=img[ma]
        if output_mask:
            return img,ma
        else:
            return img
    else:
        return img

def fs_save_surf_data(data,fname,mask=None,verbose=False):
    if mask is not None:
        # Here we assume that the data is provided as a 1D or 2D matrix and that the rows correspond to the masks indices concatenated
        if type(mask) is not list and type(mask) is not str:
            raise ValueError('Mask must be a string or a list')
        if type(fname) is not list and type(fname) is not str:
            raise ValueError('Fille name must be a string or a list')
        if type(mask) is str:
            mask=[mask]
        if type(fname) is str:
            fname=[fname]
        if len(mask)!=len(fname):
            raise ValueError('Number of output files and masks is not the same')

        nstart=0
        nstop=0
        for fout,fmask in zip(fname,mask):
            nstart=nstop
            ma,_,N=fs_load_mask(fmask)
            nstop=nstop+len(ma)
            img=np.zeros([1,N,1,data.shape[1]],dtype=float)
            img[0,ma,0,:]=data[np.arange(nstart,nstop),:]
            if verbose:
                print('Saving surface data to file ' + fout)
            nib.save(nib.Nifti1Image(img, np.eye(4)), fout)
    else:
        # Here we assume that the data is well formated as a 2D matrix, i.e. there is as many rows as surface vertices
        if data.ndim==1:
            img=np.reshape(data,[1,data.shape[0],1,1])
        elif data.ndim==2:
            img=np.reshape(data,[1,data.shape[0],1,data.shape[1]])
        else:
            img=data
        if verbose:
            print('Saving surface data to file ' + fname)
        nib.save(nib.Nifti1Image(img, np.eye(4)), fname)

def fs_surf_gradient_struct(fname,mask,verbose=False,validate_rotation=False,save_out=None):
    # Computes the gradient structure at every vertex

    if verbose:
        print('Processing '+fname)

    faces,position=fs_read_surf(fname)
    vertices=np.sort(np.unique(faces)) # Unique vertices ID
    mv=np.max(vertices)+1 # Vertices IDs are 0 based
    proj=np.empty(len(vertices),dtype=object) # Hold coordinates, within normal plane, of points onto the plane
    neigh=np.empty(len(vertices),dtype=object) # Hold indice of projected points

    # Find out which vertices are neighboring the medial wall and remove them foor cortical mask
    if verbose:
        print('Extracting cortical vertices bordering medial wall')
    cortex,medial,_=fs_load_mask(mask)
    border=np.ndarray(0,dtype=int)
    for nc in cortex:
        faces_ind=faces[np.sum(faces==nc,axis=1)>=1,:]
        in_medial=[nf in medial for nf in faces_ind]
        if np.any(in_medial):
            border=np.append(border,nc)
            neigh[nc]=faces[not in_medial]
    cortex=cortex[np.array([nc not in border for nc in cortex],dtype=bool)]

    if verbose:
        print('Computing gradient structure for cortical vertices')
    for nc,ni in zip(cortex,np.arange(0,len(cortex))):

        # Note: vertices for each faces are always ordered so that their cross product points outward,
        # hence to have consistent normals, we simply need to make sure the ordering is respected.

        if verbose:
            if ni % 10000==0 and ni != 0:
                print(str(ni) + '/' + str(len(cortex)))

        # Find the normal of each face the current vertice is part of and take the average
        faces_ind=np.where(np.sum(faces==nc,axis=1)>=1)[0] # Extract rows for each faces containing the current vertice
        fnorm=np.ndarray([len(faces_ind),3])
        for nf,nn in zip(faces_ind,len(faces_ind)):
            ind=np.where(faces[nf,:]==nc)[0]
            if ind==0:
                fnorm[nn,:]=np.cross(position[faces[nf,2],:]-position[faces[nf,0],:],
                            position[faces[nf,1],:]-position[faces[nf,0],:])
            elif ind==1:
                fnorm[nn,:]=np.cross(position[faces[nf,0],:]-position[faces[nf,1],:],
                                position[faces[nf,2],:]-position[faces[nf,1],:])
            elif ind==2:
                fnorm[nn,:]=np.cross(position[faces[nf,1],:]-position[faces[nf,2],:],
                                position[faces[nf,0],:]-position[faces[nf,2],:])

        # Take average of all faces norms and normalize final vector
        snorm=normv(normm(fnorm,axis=1).mean(axis=0))

        # Project each neighboring points onto the plane defined by the normal
        pts=np.unique(faces[faces_ind,:])
        proj_pts=np.ndarray([len(pts),3])
        px=position[pts,:]-position[nc,:] # Position of every point, ajuested for position of current vertice
        for pt in np.where(pts!=nc)[0]:
            proj_pts[pt,:]=px[pt,:]-np.dot(px[pt,:],snorm)*snorm
            proj_pts[pt,:]=proj_pts[pt,:]*(norm_arclen(px[pt,:],snorm)/vlen(proj_pts[pt,:]))

        # Find rotation of normal vector onto z-axis
        ez=np.array([0.,0.,1.])
        # Rotate along x-axis
        ax=vec_angle(ez,np.array([0,snorm[1],snorm[2]]))
        if snorm[1]<0:
            ax=-ax
        # Rotate along y-axis
        vrx,Rx=rotate3d(snorm,ax,axis=0,compute=True,matrix=True)
        ay=vec_angle(ez,vrx)
        if vrx[0]>0:
            ay=-ay
        vz,Ry=rotate3d(vrx,ay,axis=1,compute=True,matrix=True)

        # For sanity, check that snorm was well projected onto z-axis
        if validate_rotation:
            if (not math.isclose(vz[0],0.0, abs_tol=1e-9) or not math.isclose(vz[1],0.0, abs_tol=1e-9) or not
                math.isclose(vz[2],1.0, abs_tol=1e-9)):
                print(vz)
                raise ValueError('Normal vector was not well projected onto z-axis')

        # Rotate neighbor points
        proj[nc]=np.dot(np.dot(Ry,Rx),proj_pts.T).T[:,[0,1]] # 2d matrix with rows as points and columns as x and y
        neigh[nc]=pts

    if save_out is not None:
        if verbose:
            print('Saving gradient structure to '+save_out)
        np.savez(save_out,proj,neigh,cortex,border)

    return proj,neigh,cortex,border

def fs_load_surf_gradient(fname):
    # Load gradient structure
    if not fname.endswith('.npz'):
        fname=fname+'.npz'
    out=np.load(fname)
    return out['arr_0'],out['arr_1'],out['arr_2'],out['arr_3']

def fs_surf_gradient(data,fgrad,save_out=None,verbose=False):
    if verbose:
        print('Computing gradient')
    proj,neigh,cortex,border=fs_load_surf_gradient(fgrad)
    grad=np.zeros(proj.shape[0],dtype=float)

    if verbose:
        print('Processing cortical vertices')
    for nc in cortex:
        # Fit a plane to obtain the gradient
        try:
            grad[nc]=vlen(OLS(data[neigh[nc]],add_constant(proj[nc])).fit().params[1:2])
        except:
            ipdb.set_trace()
            return

    # Border vertices take the mean gradient of their neighbors
    if verbose:
        print('Processing border vertices')
    for nb in border:
        grad[nb]=grad[neigh[nb]].mean()

    if save_out is not None:
        fs_save_surf_data(grad,save_out,verbose=verbose)

    return grad

def fs_surf_view(surf,data=None,view='mid',hemi='lh',snap=False,plot_snap=False):

    mlab.init_notebook() # make plot inline in jupyter

    if plot_snap and not snap: # Make sure snapping is on if we are plotting it
        snap=True
    if view is str:
        view=[view]
    if snap:
        img=np.ndarray(len(view),dtype=object)

    for nv in np.arange(0,len(view)):
        mlab.clf()

        # Adjust view according to hemisphere being displayed
        if hemi=='lh':
            adj=0
        elif hemi=='rh':
            adj=180
        else:
            raise ValueError('Invalid hemisphere '+str(hemi))

        if data is str:
            scalars=mu.fs_load_surf_data(data)
        else:
            scalars=data

        faces,position=fs_read_surf(surf)
        mlab.triangular_mesh(position[:,0],position[:,1],position[:,2],faces,scalars=scalars)

        # Adjust view
        dist=350
        if view[nv]=='mid':
            mlab.view(azimuth=180+adj,elevation=90,distance=dist)
        elif view[nv]=='side':
            mlab.view(azimuth=0+adj,elevation=90,distance=dist)
        elif view[nv]=='back':
            mlab.view(azimuth=90,elevation=90,distance=dist)
        elif view[nv]=='front':
            mlab.view(azimuth=-90,elevation=90,distance=dist)
        else:
            raise ValueError('Invalid view '+str(view[nv]))
        if snap:
            img[nv]=mlab.screenshot()
            if not plot_snap: # in case plots are not inline
                mlab.close()
    if snap:
        if plot_snap:
            plt.figure(figsize=(8,2))
            for nv in np.arange(0,len(view)):
                plt.subplot(1,len(view),nv+1)
                plt.imshow(img[nv])
                plt.axis('off')
            plt.show()
        return img
