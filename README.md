# imagereg
This is the code for our paper https://arxiv.org/abs/1703.05264.
The method is called Total Variation Regularized Tensor-on-scalar Regression.
The proposed method and algorithm solves a regression problem with tensor as outcome and a scalar as predictor.  We encourage smoothness
for adjacent predicted means by imposing a TV regularization. Here, we provide code for implementation for 1-D 2-D and 3-D tensor outcome regression problems.

## Dependency
The package depends on the python package pygfl which is a graphical fused lasso solver, see the installation tutorial on
https://github.com/tansey/gfl.
Notice that pygfl can be installed through pip, it requires numpy, scipy, networkx, and Gnu Scientific Library gsl. 


## Tutorial
The main function of this package is in the sir3D.py  Implement.py is a implementation example code. The X and Y however is for the ADHD200 data that we cannot release the data here.

There is a distributional version for the main function TVTR
Tensor can be arbitrary dimension by user’s self-defined edges. Here we provide a function creating adjacency matrix for grid.

For example, if the outcome is 1-D time series of 500 time points, we can use the following code to create a nature adjacency matrix assumming adjacent time has similar outcome.

```
edge=creategrid(500)
```
If outcome is 2-D image of 70 by 80 pixels
```
edge=creategrid([70,80])
```

If outcome is 3-D image of 30 by 40 by30 pixels
```
edge=creatgrid([30,40,30])
```

After one created the edges, one can call the main function
```
gamma=TVTR(X,Y,edges,stacked=0,lam=1,rho=1,tol=0.05)
```

#### X  
the predictor matrix, n by p, each row is the p predictors for each subject.
#### Y 
the outcome matrix, n by m, the tensor outcome is vectorized to a m by 1 vector, each row is the vectorized tensor outcome
#### edges
the edges to assume smoothness for
#### stacked
default is 0, meaning edges is for one subject's image, if stacked=0, meaning the edges is already defined  for all n images stacked together
#### lam
the tuning parameter for the TV regularization term. The large lam is, the smoother is the estimates.
#### rho
a tuning parameter in the algorithm, the algorithm is not sensitive to this parameter.
#### tol
convergence criterian
#### gamma
The outcome gamma is the estimators. It is p by m. Each column is the vectorized smooth association map for one predictor.
