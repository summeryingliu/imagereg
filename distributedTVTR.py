import numpy as np
import math
from numpy import linalg as LA
from pygfl.easy import solve_gfl
from multiprocessing import Pool
def createpartition(y, edges,mini_batch_size=64, lam=1,rho=1):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (number of examples,input size)
    Y -- n by m
    mini_batch_size -- size of the mini-batches, integer
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    n,m=y.shape
    mini_batches = []
    num_complete_minibatches = math.floor(n / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    edge_std = edges
    for k in range(1, mini_batch_size):
        edge_std = np.vstack((edge_std, edges + m * k))

    for k in range(num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch=range(k * mini_batch_size,(k + 1) * mini_batch_size)
        minibatch = [y[mini_batch], edge_std,lam,rho]
        mini_batches.append(minibatch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if n % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        end = int(n - mini_batch_size * math.floor(n / mini_batch_size))
        mini_batch = range(num_complete_minibatches * mini_batch_size,n)
        edge_last= edges
        for k in range(1, end):
            edge_last = np.vstack((edge_last, edges + m * k))

        minibatch = [y[mini_batch],edge_last,lam,rho]
        mini_batches.append(minibatch)

    return mini_batches
#the method need minimatch to have the same size

def TVTRdistributed(X,Y,edges,mini_batch_size=16,lam=1,rho=1,tol=0.05,verbose=1):
    #y = np.array(Y, dtype='float64')
    if verbose:
        print (X.shape)
        print (Y.shape)
        print (edges.shape)

    y = np.array(Y, dtype='float64')
    m = y.shape[1]
    n= X.shape[0]
    theta = np.random.rand(n, m)
    mu = np.random.rand(n, m)
    eta = np.random.rand(n, m)
    U = np.zeros((n, m))
    V = np.zeros((n, m))
    Im = np.eye(n)
    M = Im - X.dot(LA.inv(X.T.dot(X))).dot(X.T)
    converge = 0
    iter=1

    while not converge:
        theta_last = theta
        theta = (y+rho*(mu-V+eta-U))/(1+2*rho)
        eta = (Im - M).dot((theta + U))
        ALL=createpartition(theta+V,edges,mini_batch_size,lam,rho)
        print('partition done')
        pool=Pool(8)
        mu=pool.map(multi_run_wrapper, ALL)
        mu=np.vstack(mu)
        pool.close()
        U = U + theta - eta
        V = V + theta - mu
        infeas = LA.norm(theta - eta) / LA.norm(theta)
        relerr = LA.norm(theta_last - theta) / LA.norm(theta_last)
        converge = infeas < tol and relerr < tol
        iter += 1
        print 'Iter: ' + str(iter) + '\t rel_err ' + str(relerr) + '\t Infeasibility ' + str(infeas)

    gamma = LA.inv(X.T.dot(X)).dot(X.T).dot(theta)
    return gamma

def denoisemini(y,edge,lam,rho):
    num,m=y.shape
    minibatch_mu = solve_gfl(y.reshape((num * m,), order="C"), edge, minlam=lam / rho,
                             maxlam=lam / rho, numlam=1)
    minibatch_mu=minibatch_mu.reshape((num,m))
    return minibatch_mu

def multi_run_wrapper(args):
   return denoisemini(*args)