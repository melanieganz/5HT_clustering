import numpy as np
import os

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
