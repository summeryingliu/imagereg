import numpy as np
from numpy import linalg as LA
from pygfl.easy import solve_gfl
def creategrid(dim):
    D=len(dim)
    edge=[]
    if D==1:
        for i in range(dim[0]-1):
            edge.append([i,i+1])

    if D==2:
        for i in range(dim[0]):
            for j in range(dim[1]):
                current=dim[1]*i+j
                right=current+1
                down=current+dim[1]
                if j<dim[1]-1:
                    edge.append([current,right])

                if i<dim[0]-1:
                    edge.append([current,down])


    if D==3:
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    current=dim[1]*dim[2]*i+dim[2]*j+k
                    back=current+1
                    down=dim[1]*dim[2]*(i+1)+dim[2]*j+k
                    right=dim[1]*dim[2]*i+dim[2]*(j+1)+k
                    if k<dim[2]-1:
                        edge.append([current,back])

                    if i<dim[0]-1:
                        edge.append([current,down])

                    if j<dim[1]-1:
                        edge.append([current,right])

    edges=np.asarray(edge,dtype='int')
    return edges

#this function takes vectorized y and x
#the dimension does not matter here
def TVTR(X,Y,edges,stacked=0,lam=1,rho=1,tol=0.05,verbose=1):
    #y = np.array(Y, dtype='float64')
    print (X.shape)
    print (Y.shape)
    print (edges.shape)
    y = np.array(Y, dtype='float64')
    n=X.shape[0]
    m=Y.shape[1]
    Im = np.eye(n)
    M = Im - X.dot(LA.inv(X.T.dot(X))).dot(X.T)
    edge_all=edges
    if stacked==0:
        for k in range(1, n):
            edge_all = np.vstack((edge_all, edges + m * k))

    edge_all = np.asarray(edge_all, dtype='int')
    converge = 0
    #initialize parameters
    theta = np.random.rand(n, m)
    mu = np.random.rand(n, m)
    eta = np.random.rand(n, m)
    U = np.zeros((n, m))
    V = np.zeros((n, m))
    # infeas = LA.norm(M.dot(G))
    iter = 0
    while not converge:
        theta_last = theta
        print(iter)
        theta = (y+rho*(mu-V+eta-U))/(1+2*rho)
        eta = (Im - M).dot((theta + U))
        #print(np.shape(theta+V))
        if verbose:
            print('entering interation '+str(iter)+' solve_gfl')

        mu = solve_gfl((theta+V).reshape((n*m,), order="C"), edge_all, minlam=lam / rho, maxlam=lam / rho, numlam=1)
        mu = mu.reshape(n, m)
        U = U + theta - eta
        V = V + theta - mu
        infeas = LA.norm(theta - eta) / LA.norm(theta)
        relerr = LA.norm(theta_last - theta) / LA.norm(theta_last)
        converge = infeas < tol and relerr < tol
        iter += 1
        print('Iter: ' + str(iter) + '\t rel_err ' + str(relerr) + '\t Infeasibility ' + str(infeas))

    gamma = LA.inv(X.T.dot(X)).dot(X.T).dot(theta)
    return gamma





