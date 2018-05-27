import numpy as np
from sir3D import creategrid
from minibatchTVTR import random_mini_batches
from minibatchTVTR import TVTRminibatch
from numpy import linalg as LA
from pygfl.easy import solve_gfl
#y=np.genfromtxt('/home/summer/gfl/Y.csv',delimiter=',')
#y=np.genfromtxt('/home/summer/gfl/Y303630.csv',delimiter=',')
y=np.genfromtxt('/home/summer/gfl/Y303630.csv',delimiter=',')
x=np.genfromtxt('/home/summer/gfl/X.csv',delimiter=',')
x=x-np.mean(x,axis=0)
y=y-np.mean(y,axis=0)
assert(isinstance(x,np.ndarray))
assert(isinstance(y,np.ndarray))
#dim=[12,16,12]
dim=[30,36,30]
edge=creategrid(dim)
gamma=TVTRminibatch(x,y,edge,lam=0.5,mini_batch_size=8,num_epochs=20)
np.savetxt('gamma_minibatch303630.csv',gamma,delimiter=',')