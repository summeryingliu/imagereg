import numpy as np
import math
from numpy import linalg as LA
from pygfl.easy import solve_gfl
def random_mini_batches(X, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (number of examples,input size)
    Y -- n by m
    mini_batch_size -- size of the mini-batches, integer
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    n = X.shape[0]  # number of training examples
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(n))
    #shuffled_X = X[permutation,:]
    #shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(n/mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches=int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        #mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size,:]
        #mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size,:]
        ### END CODE HERE ###
        mini_batch = permutation[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if n % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        #end = n - mini_batch_size * math.floor(n / mini_batch_size)
        #mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:,:]
        #mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:,:]
        ### END CODE HERE ###
        mini_batch = permutation[num_complete_minibatches * mini_batch_size:]
        mini_batches.append(mini_batch)

    return mini_batches

#the method need minimatch to have the same size
def TVTRminibatch(X,Y,edges,num_epochs=1,mini_batch_size=16,lam=1,rho=1,tol=0.05,verbose=1,seed=135):
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
    for i in range(num_epochs):
        seed=seed+1
        iter = 1
        minibatches=random_mini_batches(X,mini_batch_size,seed)
        for minibatch in minibatches:
            theta_last = theta
            theta = (y+rho*(mu-V+eta-U))/(1+2*rho)
            eta = (Im - M).dot((theta + U))
        #print(np.shape(theta+V))
            print('entering epoch '+str(i)+ ' , iteration' + str(iter)+ ' solve_gfl')

            n_minibatch=len(minibatch)
            edge_all = edges
            for k in range(1, n_minibatch):
                edge_all = np.vstack((edge_all, edges + m * k))

            minibatch_mu = solve_gfl((theta+V)[minibatch].reshape((n_minibatch*m,), order="C"), edge_all, minlam=lam / rho, maxlam=lam / rho, numlam=1)
            mu[minibatch] = minibatch_mu.reshape(n_minibatch, m)
            U = U + theta - eta
            V = V + theta - mu
            infeas = LA.norm(theta - eta) / LA.norm(theta)
            relerr = LA.norm(theta_last - theta) / LA.norm(theta_last)
            converge = infeas < tol and relerr < tol
            iter += 1
            print 'epoch '+str(i)+ 'Iter: ' + str(iter) + '\t rel_err ' + str(relerr) + '\t Infeasibility ' + str(infeas)
            if converge==1:
                break

    gamma = LA.inv(X.T.dot(X)).dot(X.T).dot(theta)
    return gamma

