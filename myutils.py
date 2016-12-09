import numpy as np
import os
import struct
from statsmodels.api import OLS,add_constant
from os.path import join as opj
import math

def balance_svd_components(trg,src,return_balanced=False):
    nt=np.shape(trg)[1]
    ns=src.shape[1]
    trg_ind=np.arange(0,nt)
    src_ind=np.zeros(ns,dtype=int)
    flip_ind=np.ones(ns,dtype=float)
    for nk in np.arange(0,nt):
        cc=[np.dot(np.transpose(src[:,nk]),trg[:,ct]) for ct in trg_ind]
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
    return np.arccos(np.dot(v1/L2_norm(v1), v2/L2_norm(v2)))

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

def L2_norm(x,axis=-1):
    return np.sum(np.abs(x)**2,axis=axis)**(1./2)

def normv(v):
    return v/L2_norm(v)

def normm(data,axis=-1):
    return np.apply_along_axis(normv,axis,data)

# FreeSurfer stuff

def fs_read_surf(fname,verbose=False):

# This function is derived from freesurfer_read_surf.m
# Note: files are encoded in big-endian

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
        vertices=np.reshape(struct.unpack('>' + str(3*nf) + 'i',f.read(nf*12)),[nf,3],order='C') # 3 * int32

    f.close()
    return vertices,position

def fs_read_label(fname):
    with open(fname) as f:
        for i,line in enumerate(f): # i starts at 0
            if i==1: # Line 2 contains the number of vertices in the label
                nv=np.zeros(int(line[:-1]),dtype=int)
            elif i>1: # Two first lines are header
                nv[i-3]=int(line.split()[0])
    f.close()
    return nv

def fs_surf_neighborhood(fname,out_type='matrix',verbose=False,save_out=''):
    
    if out_type not in {'matrix','list'}:
        raise ValueError('Invalid out_type ' + out_type)
    
    # Build neighborhood matrix where neigh(i,j)=True is i and j are neighbors, False otherwise.
    if verbose:
        print('Reading ' + fname)
    vertices,_=fs_read_surf(fname)
    verticesID=np.unique(vertices) # Unique vertices ID
    mv=np.max(vertices)+1 # Vertices IDs are 0 based
    
    # Preallocate memory
    if out_type=='list':
        neigh=np.ndarray(mv,dtype=object)
    else:
        neigh=np.zeros([mv,mv],dtype=bool)
    
    # Identify neighbors
    if out_type=='list': # This is faster than matrix
        for nv in verticesID:
            if verbose and nv % 10000==0 and nv != 0:
                print(str(nv) + '/' + str(len(verticesID)))
            neighID=np.unique(vertices[np.sum(vertices==nv,axis=1)>=1,:])
            neigh[nv]=neighID[neighID!=nv]
    else:
        for nv in verticesID:
            if verbose and nv % 10000==0 and nv != 0:
                print(str(nv) + '/' + str(len(verticesID)))
            neigh[nv,vertices[np.sum(vertices==nv,axis=1)>=1,:]]=True # Preliminary testing shows that it is faster to not take unique column indices
        neigh[np.eye(mv,dtype=bool)]=False # A vertice is not neighbor is itself
    
    if len(save_out)>0:
        if verbose:
            print('Saving file ' + save_out + '.' + out_type)
        np.save(save_out + '.' + out_type,neigh)
    
    return neigh

def fs_create_cortex_mask(subjects_dir,targ,hemi,validate=True,verbose=False,save_out=''):
    
    cortex=fs_read_label(opj(subjects_dir,targ,'label',hemi+'.cortex.label'))
    
    # Make sure each cortex vertice is neighbor of at least another cortex vertice
    # This sanity check is necessary since this vertices of the medial wall have previously
    # been labeled as cortex for fsaverage, left hemisphere (e.g. fsaverage/surf/lh.pial)
    if validate:
        neigh=fs_surf_neighborhood(opj(subjects_dir,targ,'surf',hemi+'.pial'),out_type='list',verbose=verbose) 
        
        if verbose:
            print('Validating cortex mask')
            ni=0        
        invalid=np.ndarray(0,dtype=int)
        for nc in cortex:
            if verbose:        
                if ni % 10000==0 and ni != 0:
                    print(str(ni) + '/' + str(len(cortex)))
                ni=ni+1
            if np.sum([vert in cortex for vert in neigh[nc]])<2: # Require at least two neighbors in cortex
                if verbose:
                    print('Detected invalid vertex ' + str(nc))
                np.append(invalid,nc)
        cortex=cortex[np.array([vert not in invalid for vert in cortex],dtype=bool)]
    
    if len(save_out)>0:
        if verbose:
            print('Saving file ' + save_out)
        np.savez(save_out,cortex)
    
    return cortex

def fs_load_mask(fname):
    mask=np.load(fname)
    return mask['arr_0']

def fs_load_surf_data(fname,fmask=''):
    img=nib.load(fname)
    img=np.squeeze(img.get_data())
    if len(fmask)>0:
        if not fmask.endswith('.npz'):
            fmask=fmask+'.npz'
        mask=fs_load_mask(fmask)
        
        if img.ndim>1 and img.shape[1]>1:
            return img[mask,:],mask
        else:
            return img[mask],mask
    else:
        return img

def fs_save_surf_data(data,fname,fmask='',verbose=True):
    if len(fmask)>0:
        mask=fs_load_mask(fmask)
        img[0,mask,0,:]=img
    elif img.ndim==2:
        img=reshape(img,[1,img.shape[0],1,img.shape[1]])
    # Assume that at this point the data is well formated
    if verbose:
        print('Saving data with dimensions ' + str(data.shape) + ' to file ' + fname)
    nib.save(nib.Nifti1Image(img, np.eye(4)), fname)

def fs_surf_gradient_struct(fname,verbose=False,validate_rotation=False):
    # Computes the average normal at every vertices
        
    faces,position=fs_read_surf(fname)
    vertices=np.sort(np.unique(faces)( # Unique vertices ID
    mv=np.max(vertices)+1 # Vertices IDs are 0 based
    proj=np.ndarray(len(vertices),dtype=object) # Hold coordinates, within normal plane, of points onto the plane
    neigh=np.ndarray(len(vertices),dtype=object) # Hold indice of projected points
    
    if verbose:
        print('Computing gradient structure at every vertice')
    for nv in vertices:
        
        # Note: vertices for each faces are always ordered so that their cross product points outward,
        # hence to have consistent normals, we simply need to make sure the ordering is respected.
        
        if verbose and nv % 10000==0 and nv != 0:
                print(str(nv) + '/' + str(len(vertices)))
        
        # Find the normal of each face the current vertice is part of and take the average
        face_vertices=np.where(np.sum(vertices==nv,axis=1)>=1)[0] # Extract rows for each faces containing the current vertice
        fnorm=np.ndarray([len(faces),3])
        nn=0
        for nf in faces:    
            ind=np.where(vertices[nf,:]==nv)[0]
            if ind==0:
                fnorm[nn,:]=np.cross(position[vertices[nf,2],:]-position[vertices[nf,0],:],
                            position[vertices[nf,1],:]-position[vertices[nf,0],:])
            elif ind==1:
                fnorm[nn,:]=np.cross(position[vertices[nf,0],:]-position[vertices[nf,1],:],
                                position[vertices[nf,2],:]-position[vertices[nf,1],:])
            elif ind==2:
                fnorm[nn,:]=np.cross(position[vertices[nf,1],:]-position[vertices[nf,2],:],
                                position[vertices[nf,0],:]-position[vertices[nf,2],:])

            nn=nn+1   

        # Take average of all faces norms and normalize final vector
        snorm=normv(normm(fnorm,axis=1).mean(axis=0))
        
        # Project each neighboring points onto the plane defined by the normal
        pts=np.unique(vertices[faces,:])
        proj_pts=np.ndarray([len(pts),3])
        px=position[pts,:]-position[nv,:]
        for pt in np.arange(0,len(pts)):            
            proj_pts[pt,:]=px[pt,:]-np.dot(px[pt,:],snorm)*snorm
        
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
        proj[nv]=np.transpose(np.dot(np.dot(Ry,Rx),np.transpose(proj_pts)))[:,[0,1]] # 2d matrix with rows as points and columns as x and y
        neigh[nv]=pts
        
    return proj,neigh

def build_surface_gradient_matrix(fname,verbose=False):

    snorm=fs_surf_vertices_normals(fname,verbose=verbose)
    
    # For every vertice, find orthogonal plane to normal, projects neigbors and compute coordinates in plane
    for nv in vertices:        
        # Project each point onto the plane defined by the normal vector        
        u1=normalize([1,1,-(snorm[nv,0]+snorm[nv,1])/snorm[nv,2]]) # v1*u1+v2*u2+v3*u3=0 -> v1+v2+v3*u3=0 -> u3=-(v1+v2)/v3
        u2=normalize(np.cross(u1,v))

    # Load vertices forming each triangle in the surface mesh
    N,position=build_neighborhood_matrix(fname)
    gradvec=np.zeros([N.shape[0],3])
    for nv in np.arange(0,N.shape[0]):
        # Project each neighbor on normal plan

        # Adjust position of each neighbor by making sure that the distance is
        # equal to the arc length

        # Fit a plane to obtain the gradient
        gradvec[nv,:]=OLS(data[N[:,nv]],add_constant(position[N[nv,:]])).fit().params[1:4]

    return gradvec
