import pymaster as nmt
import healpy as hp
import numpy as np

# ali_ma_512 = hp.read_map("/smc/jianyao/Ali_maps/ali_mask_wo_edge_512.fits", verbose=False)

# beam_low = 27.9435; nside = 512; lmax = 1000; ell_n = b.; eln2 = utils.l2(ell_n)
    
class Pseudo_Cl(object):
    
    def __init__(self, mask_in, nside, bin_w, lmax, beam = None):
        
        '''
        Define the **apodized mask**, **beam weights**, **nside**, **bin-scheme**, **ell**
        
        Needs to be revised for the beam correction
        '''
        self.mask = nmt.mask_apodization(mask_in, 6, apotype='C2')
        
        self.nside = nside; self.lmax = lmax
        
#         self.beam = hp.gauss_beam(beam/60/180*np.pi, lmax = 3*self.nside); 
        
        self.b = nmt.NmtBin(self.nside, nlb=bin_w, lmax=self.lmax, is_Dell = True)
        
        self.ell_n = self.b.get_effective_ells(); self.lbin = len(self.ell_n)



    def compute_master(self, f_a, f_b, wsp):
        
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        
        return cl_decoupled

    def Cross_TT(self, maps):
    
        ''' 
        revised for upper-triangle of the matrix 

        Input: spin-1 maps; Only multi-frequency TT maps; Or BB maps by LH's method;
        This function doesn't need *purify_B = True*
        '''
        
        map0 = np.ones(12*self.nside**2)
        m0 = nmt.NmtField(self.mask, [map0])#, beam=bl); to make len(map) equal 1. 

        # construct a workspace that calculate the coupling matrix first.
        _w = nmt.NmtWorkspace()
        _w.compute_coupling_matrix(m0, m0, self.b)
        
        n_f = len(maps);
        cl = np.zeros((n_f*n_f, self.lbin)); Cl = np.zeros((self.lbin, n_f, n_f))

        k = 0
        for i in range(n_f):
            for j in range(n_f):
                if i >= j :
                    
                    m_i = nmt.NmtField(self.mask, [maps[i]])#beam=bl); # BB maps at i-th fre
                    m_j = nmt.NmtField(self.mask, [maps[j]])#beam=bl); # BB maps at j-th fre
                    
                    cl[k] = self.compute_master(m_i, m_j, _w)[0]
                else:
                    cl[k] = np.zeros(self.lbin)
                k += 1

        for l in range(self.lbin):
            Cl[l, 0:n_f , 0:n_f] = cl[:,l].reshape(n_f, n_f)
            Cl[l] += Cl[l].T - np.diag(Cl[l].diagonal()) 
            
        return Cl

    def Cross_EB(self, maps):

        '''
        Calculate the E- and B-mode power spectrum utilize Namaster purify_B method.

        Given parameters:
        ----------------
        maps : input maps with IQU component. Only Q and U are needed in this EB estimation. maps[i][1:]
        ell_n : the effective number of l_bins
        mask : apodized mask 
        beam : the gaussian beam weights for each multipole

        '''

        
        # - To construct a empty template with a mask to calculate the **coupling matrix**

        # > the template should have a shape (2, 12*nside^2)

        map0 = np.ones((2, 12*self.nside**2))
        m0 = nmt.NmtField(self.mask, map0, purify_e=False, purify_b=True)#, beam=bl)

        # construct a workspace that calculate the coupling matrix first.
        _w = nmt.NmtWorkspace()
        _w.compute_coupling_matrix(m0, m0, self.b)
    
        n_f = len(maps); 
        cl = np.ones((2, n_f*n_f, self.lbin)); Cl = np.zeros((2, self.lbin, n_f, n_f))
        k = 0
        for i in range(n_f):
            for j in range(n_f):

                if i >= j :

                    m_i = nmt.NmtField(self.mask, maps[i][1:], purify_e=False, purify_b=True)#beam=bl); #Q and U maps at i-th fre
                    m_j = nmt.NmtField(self.mask, maps[j][1:], purify_e=False, purify_b=True)#beam=bl); #Q and U maps at j-th fre

                    cross_ps = self.compute_master(m_i, m_j, _w) ## EE, EB, BE, BB
                else:
                    cross_ps = np.zeros((4, self.lbin)) 

                cl[0][k] = cross_ps[0]; cl[1][k] = cross_ps[3]  ## assign the E and B_mode power spectrum 
                k += 1

        for l in range(self.lbin):
            Cl[0, l, : , :] = cl[0, :,l].reshape(n_f, n_f); Cl[1, l, : , :] = cl[1, :,l].reshape(n_f, n_f)
            Cl[0, l] += Cl[0, l].T - np.diag(Cl[0, l].diagonal()) ; Cl[1, l] += Cl[1, l].T - np.diag(Cl[1, l].diagonal()) 
            
        return Cl