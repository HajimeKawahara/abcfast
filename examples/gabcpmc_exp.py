from abcfast.gabcpmc import *
from abcfast.utils import statutils
       
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from scipy.stats import gamma as gammafunc
    from scipy.stats import norm as normfunc
    import time
    import sys
    
    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC PMC Method.")
    print("This code demonstrates an exponential example in Section 5 in Turner and Van Zandt (2012) JMP 56, 69, with some modifications.")
    print("*******************************************")

    #preparing data
    nsample=500
    lambda_true=0.1
    Yobs=random.exponential(1.0/lambda_true,nsample)

    # start ABCpmc 
    abc=ABCpmc()
    abc.maxtryx=10000000
    abc.npart=1024
    abc.wide=2.0
    # input model/prior
    abc.nparam=1
    abc.model=\
    """
    /* the exponential distribution model generator */

    __device__ void model(float* Ysim, float* param, curandState* s, float* aux, int isample){
    
    Ysim[0] = -log(curand_uniform(s))/param[0];

    }
    """
    
    # prior 
    alpha0=0.1
    beta0=0.1
    def fprior():
        def f(x):
            return gammafunc.pdf(x, alpha0,scale=1.0/beta0)
        return f
    abc.fprior = fprior()#

    abc.prior=\
    """
    #include "gengamma.h"

    __device__ void prior(float* param,curandState* s){

    param[0] = gammaf(0.1,0.1,s);

    return;

    }
    """

    # data and the summary statistics
    abc.nsample = len(Yobs)
    abc.ndata = 1
    Ysum = np.sum(Yobs)
    abc.Ysm = np.array([Ysum])
    
    #set prior parameters
    abc.epsilon_list = np.array([3.0,1.0,1.e-1,1.e-2,1.e-3,1.e-4])

    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()
#    plt.hist(abc.x,bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    tend = time.time()

    print(tend-tstart,"sec")
    
    #plotting...
    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(211)
    ax.hist(abc.x,bins=30,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    ax.hist(abc.xres(),bins=30,label="resampled",density=True,alpha=0.2)

    alpha=alpha0+abc.nsample
    beta=beta0+Ysum
    xl = np.linspace(gammafunc.ppf(0.0001, alpha,scale=1.0/beta),gammafunc.ppf(0.9999, alpha,scale=1.0/beta), 100)
    ax.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\pi_\mathrm{ABC}$")
    plt.legend()
    ax=fig.add_subplot(212)
    ax.plot(abc.x,abc.w,".")
    plt.xlabel("$\lambda$")
    plt.ylabel("$weight$")
    plt.savefig("abcpmc.png")
    plt.show()

