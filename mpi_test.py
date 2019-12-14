from mpi4py import MPI 
import healpy as hp
import numpy as np

def Mask(maps): 
    maps_ma = hp.ma(maps)
    maps_ma.mask = np.logical_not(ali_ma)
    return maps_ma

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## generate noise maps 

if rank == 0:
#    data = np.ones(10)                                                              
    data = hp.read_map('/smc/jianyao/Ali_maps/Noise_maps/fits/I_Noise_95_G_1024.fits', partial=True)
#    data = hp.read_map('/smc/jianyao/Ali_maps/Noise_maps/fits/I_Noise_95_G_1024.fits', partial=True)                                                      
#    print("process {} bcast data {} to other processes".format(rank, data))
else:                                                                      
    data = None 

data = comm.bcast(data,root=0)
n2zeros = data*0 #0,0,nan,nan......
index = np.arange(len(data))

nIQU = np.zeros((3,len(data)))

#a = np.random.rand()
for i in index[n2zeros==0]:    #full_ali map, without mask out the edge.
    nIQU[0][i] = np.random.normal(0, data[i])
    nIQU[1][i] = np.random.normal(0, data[i]*np.sqrt(2))
    nIQU[2][i] = np.random.normal(0, data[i]*np.sqrt(2))

#print(nIQU)
hp.write_map('/smc/jianyao/Ali_maps/mpitest/noise_realizations_95GHz_%s.fits'%(rank), nIQU)

'''
### calculate the mean value and standard deviation of the noise power spectrum from different noise realizations.

lmax = 2000

noise_map = hp.read_map('/smc/jianyao/Ali_maps/Noise_realizations/95GHz/noise_realizations_95GHz_%s.fits'%(rank), field = None)

ali_ma = hp.read_map('/smc/jianyao/Ali_maps/ali_mask_wo_edge.fits')

noise_mask = Mask(noise_map)

nl = hp.anafast(noise_mask, nspec = 3, lmax = lmax)
if rank == 0:

	noise_all = np.zeros((size, 3, lmax + 1))
	for i in range(1,size):
		noise_all[i] = comm.recv(source = i#)
	np.save('nl_al.npy', noise_all)
else:
	comm.send(nl, dest = 0)

#noise_all = comm.gather(nl, root = 0)
#np.save('nl_al1.npy', noise_all)

'''
