'''
This module contains some basic helper functions.
2019-12-20
With Yun Zheng
'''

import numpy as np
import healpy as hp

def bin_l(cl, L, Q, l2 = False):
    ''' 
    
    cl, L, Q(bin_number); no np.mean
    
    '''
    if len(cl.shape) > 2 :
        bin_averages = np.zeros((Q, cl.shape[1], cl.shape[1]))
    else:
        bin_averages = np.zeros(Q)
    if l2:
        for l in range(L):
            cl[l] = l*(l+1)/2/np.pi*(cl[l])    
    for q in range(Q):
        if q == 0:
            bin_averages[q] = sum(cl[2:((q+1)*L//Q)]/(L//Q))   ### l = 0,1 should be abandoned; SHOULDN'T use np.mean(), becaues it need to specify AXIS... . But for different shape, the axis is different .
        else:
            bin_averages[q] = sum(cl[q*L//Q:((q+1)*L//Q)]/(L//Q))
    return bin_averages

def get_ell(L, Q):
    ''' L, Q(bin_number) '''
    Ell = np.ones(Q)
    for i in range(0,int(Q)):
        Ell[i] = (2*i+1)*L//Q/2 
    return Ell

def Power_spectrum(maps,R, lmax):
    global cl
    ''' 
    revised for upper-triangle of the matrix.
    Input:
    maps with multi-frequencies IQU sky maps. 
    Galactic plane cut for calculating the power spectrum.
    lamx.  

    need to be generalized to E B PS simulantaneously. 
    '''
    n_f = len(maps)
    cl = np.ones((n_f*n_f, lmax +1)); Cl = np.zeros((lmax+1, n_f, n_f))
    k = 0
    for i in range(n_f):
        for j in range(n_f):
            
            if i >= j :
                cross_ps = hp.anafast(maps[i], maps[j], lmax = lmax, gal_cut=R, nspec=3)
            else:
                cross_ps = np.zeros((3, lmax+1)) ## TT, EE, BB
            cl[k] = cross_ps[2]  ## calculate the B_mode power spectrum 
            k += 1
    for l in range(lmax+1):
        Cl[l, 0:n_f , 0:n_f] = cl[:,l].reshape(n_f, n_f)
        Cl[l] += Cl[l].T - np.diag(Cl[l].diagonal()) 
    return Cl

def m_l(lmax, l):
    ''' 
    Return the m of corresponding l.
    
    '''
    m_id = np.ones(l+1, dtype = np.int)
    for i in range(l + 1):
        m_id[i] = hp.sphtfunc.Alm.getidx(lmax, l , i)
    return (m_id)


ali_ma = hp.read_map('/smc/jianyao/Ali_maps/ali_mask_wo_edge.fits')#, verbose=False)
def Mask(maps): 
    maps_ma = hp.ma(maps)
    maps_ma.mask = np.logical_not(ali_ma)
    return maps_ma

def l2(ell):
    '''
    get the l^2/np.pi
    '''
    
    return ell*(ell+1)/2/np.pi

