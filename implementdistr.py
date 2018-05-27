import numpy as np
import timeit
start = timeit.timeit()
from sir3D import creategrid
from distributedTVTR import TVTRdistributed
from distributedTVTR import createpartition
from distributedTVTR import *
from multiprocessing import Pool
from numpy import linalg as LA
from pygfl.easy import solve_gfl
#y=np.genfromtxt('/home/summer/gfl/Y.csv',delimiter=',')
#y=np.genfromtxt('/home/summer/gfl/Y121612.csv',delimiter=',')
y=np.genfromtxt('/home/summer/gfl/Y303630.csv',delimiter=',')
x=np.genfromtxt('/home/summer/gfl/X.csv',delimiter=',')
x=x-np.mean(x,axis=0)
y=y-np.mean(y,axis=0)
assert(isinstance(x,np.ndarray))
assert(isinstance(y,np.ndarray))
#dim=[12,16,12]
dim=[30,36,30]
edge=creategrid(dim)
gamma=TVTRdistributed(x,y,edge,lam=0.05,mini_batch_size=8)
np.savetxt('gamma_dist303630.csv',gamma,delimiter=',')
end = timeit.timeit()
print end - start