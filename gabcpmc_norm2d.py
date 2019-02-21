from gabc.gabcpmc import *
from gabc.utils.statutils import *
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from scipy.stats import multivariate_normal as mnormfunc
    import time
    import sys
    
    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC PMC Method.")
    print("This code demonstrates a 2 dimensional normal distribution by Akeret+2015")
    print("*******************************************")


    
    #data
    n=1000
    sigma = np.eye(2) * 0.25
    means = [0.7, 1.5]
    Yobs=random.multivariate_normal(means, sigma, n)

    #start ABCpmc 
    abc=ABCpmc()

    abc.maxtryx=1000000#debug magic
    abc.npart=512#debug magic
    
    # input model/prior
    abc.nmodel=2 #number of the model parameter = 2: (mean0, mean1)
    abc.model=\
    """
    #include "gennorm.h"

    /* the exponential distribution model generator */

    __device__ float model(float* par, float* xmodel, curandState* s){

    xmodel[0] = gennormf(par[0],0.25,s);
    xmodel[1] = gennormf(par[1],0.25,s);

    }
    """

    
    abc.prior=\
    """

    __device__ void prior(float* parprior,float* xprior,curandState* s){

    xprior[0] = gennormf(parprior[0],parprior[1],s);
    xprior[1] = gennormf(parprior[2],parprior[3],s);

    return;

    }
    """

    # data and the summary statistics
    abc.n = len(Yobs)
    abc.ndata = 2 #data dimension
    Ysum = np.sum(Yobs,axis=0)
    abc.Ysm = Ysum
    
    # prior functional form
    def fprior():
        def f(x,parprior):
            means = [parprior[0], parprior[2]]
            covs  = np.matrix([[parprior[1], 0.0], [0.0, parprior[3]]])
            return mnormfunc.pdf(x, mean=means, cov=covs)
        return f
    abc.fprior = fprior()#

    
    #set prior parameters
    abc.parprior=np.array([1.0,0.5,1.0,0.5]) #fixed parameters of a prior, mean0, sigma0, mean1, sigma1.
    
    abc.epsilon_list = np.array([1.e-1,1.e-2,1.e-3,1.e-4])

    #initial run of abc pmc
    abc.run()
    abc.check()
    xw=np.copy(abc.x).reshape(abc.npart,abc.nmodel)
    plt.hist(xw[:,0],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5,color="C0")
    plt.hist(xw[:,1],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5,color="C1")
    plt.axvline(means[0],color="C0")
    plt.axvline(means[1],color="C1")
    plt.show()

    sys.exit("ToDo: to match comp_weight for nmodel>1.")
    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    #plotting...
    xw=abc.x.reshape(abc.npart,abc.nmodel)
#    print(np.shape(xw))
#    plt.hist(xw[:,0],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
#    plt.hist(xw[:,1],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    plt.show()
