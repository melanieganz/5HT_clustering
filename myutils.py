import numpy as np
import os
import struct
from statsmodels.api import OLS,add_constant

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

def freesurfer_read_surf(fname,verbose=False):

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

def build_neighborhood_matrix(fname):

    vertices,position=freesurfer_read_surf(fname)

    verticesID=np.unique(vertices) # Unique vertices ID

    mv=np.max(vertices)+1 # Vertices IDs are 0 based
    N=np.zeros([mv,mv],dtype=bool)
    for nv in verticesID:
        N[nv,np.unique(vertices[np.sum(vertices==nv,axis=1)>=1,:])]=True

    N[np.eye(mv,dtype=bool)]=False
    return N,position

def build_surface_gradient_matrix(data,fname):

    # Load vectices position

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
