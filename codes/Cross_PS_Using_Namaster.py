import pymaster as nmt
import healpy as hp

# Define the **apodized mask**, **beam weights**, **nside**, **bin-scheme**, **ell**

beam_low = 27.9435; nside = 512; lmax = 1000

b = nmt.NmtBin(nside, nlb=10, lmax=lmax); ell_n = b.get_effective_ells(); eln2 = utils.l2(ell_n)

bl = hp.gauss_beam(beam_low/60/180*np.pi, lmax = 3*nside)

ali_ma_512 = hp.read_map("/smc/jianyao/Ali_maps/ali_mask_wo_edge_512.fits", verbose=False)
mask = nmt.mask_apodization(ali_ma_512, 6, apotype='C2')

def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

# - To construct a empty template with a mask to calculate the **coupling matrix**

# > the template should have a shape (2, 12*nside^2)

map0 = np.ones((2, 12*nside**2))
m0 = nmt.NmtField(mask, map0, purify_e=False, purify_b=True, beam=bl)

# construct a workspace that calculate the coupling matrix first.
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(m0, m0, b)

def Cross_PS(maps):
    
    '''
    Calculate the E- and B-mode power spectrum utilize Namaster purify_B method.
    
    Given parameters:
    ----------------
    ell_n : the effective number of l_bins
    mask : apodized mask 
    beam : the gaussian beam weights for each multipole
        
    '''
    
    n_f = len(maps); lbin = len(ell_n)
    cl = np.ones((2, n_f*n_f, lbin)); Cl = np.zeros((2, lbin, n_f, n_f))
    k = 0
    for i in range(n_f):
        for j in range(n_f):
            
            if i >= j :
                
                m_i = nmt.NmtField(mask, maps[i][1:], purify_e=False, purify_b=True)#beam=bl); #Q and U maps at i-th fre
                m_j = nmt.NmtField(mask, maps[j][1:], purify_e=False, purify_b=True)#beam=bl); #Q and U maps at j-th fre
                
                cross_ps = compute_master(m_i, m_j, w) ## EE, EB, BE, BB
            else:
                cross_ps = np.zeros((4, len(ell_n))) 
                
            cl[0][k] = cross_ps[0]; cl[1][k] = cross_ps[3]  ## assign the E and B_mode power spectrum 
            k += 1
            
    for l in range(lbin):
        Cl[0, l, : , :] = cl[0, :,l].reshape(n_f, n_f); Cl[1, l, : , :] = cl[1, :,l].reshape(n_f, n_f)
        Cl[0, l] += Cl[0, l].T - np.diag(Cl[0, l].diagonal()) ; Cl[1, l] += Cl[1, l].T - np.diag(Cl[1, l].diagonal()) 
    return Cl