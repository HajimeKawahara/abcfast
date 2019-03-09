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
    nsample=1000
    sigma = np.eye(2) * 0.25
    means = [1.1, 1.5]
    Yobs=random.multivariate_normal(means, sigma, nsample)

    #start ABCpmc 
    abc=ABCpmc()

    abc.maxtryx=1000000#debug magic
    abc.npart=512#debug magic
    
    # input model/prior
    abc.nparam=2 #number of the model parameter = 2: (mean0, mean1)
    abc.model=\
    """
    #include "gennorm.h"

    /* the exponential distribution model generator */

    __device__ void model(float* Ysim,float* param,  curandState* s){

    Ysim[0] = normf(param[0],0.25,s);
    Ysim[1] = normf(param[1],0.25,s);

    }
    """
    #set prior
    #fixed parameters of a prior, mean0, sigma0, mean1, sigma1.
    hparam=np.array([1.0,0.5,1.0,0.5])
    
    # prior functional form
    def fprior():
        def f(x):
            means = [hparam[0], hparam[2]]
            covs  = np.matrix([[hparam[1], 0.0], [0.0, hparam[3]]])
            return mnormfunc.pdf(x, mean=means, cov=covs)
        return f
    abc.fprior = fprior()#
    
    abc.prior=\
    """

    __device__ void prior(float* param,curandState* s){

    param[0] = normf(1.0,0.5,s);
    param[1] = normf(1.0,0.5,s);

    return;

    }
    """

    # data and the summary statistics
    abc.nsample = len(Yobs)
    abc.ndata = 2 #data dimension
    Ysum = np.sum(Yobs,axis=0)
    abc.Ysm = Ysum

    abc.epsilon_list = np.array([1.0,0.5,0.3,0.1,0.05])

    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()
    xw0=np.copy(abc.xw)


    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    #plotting...

    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.hist(xw0[:,0],bins=20,label="$\epsilon$="+str(abc.epsilon_list[0]),density=True,alpha=0.4,color="C0")
    ax.hist(xw0[:,1],bins=20,label="$\epsilon$="+str(abc.epsilon_list[0]),density=True,alpha=0.4,color="C1")

    ax.axvline(means[0],color="C0")
    ax.axvline(means[1],color="C1")
    ax.hist(abc.xw[:,0],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.7,color="C0")
    ax.hist(abc.xw[:,1],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.7,color="C1")
    ax.legend()

    ax2=fig.add_subplot(122)
#    ax2.plot(xw0[:,0],xw0[:,1],".",alpha=0.3)
    cl=ax2.scatter(abc.xw[:,0],abc.xw[:,1],c=abc.w,alpha=0.5)
    plt.colorbar(cl)
    ax2.plot([means[0]],[means[1]],"s",color="red")
    plt.show()
