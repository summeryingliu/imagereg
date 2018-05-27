import numpy as np
from sir3D import creategrid
from sir3D import TVTR
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
gamma=TVTR(x,y,edge,lam=0.1)
np.savetxt('gamma_test303630.csv',gamma,delimiter=',')