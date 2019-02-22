from gabc.gabcpmc import *
from gabc.utils.statutils import *
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from scipy.stats import gamma as gammafunc
    import time
    import sys
    
    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC PMC Method.")
    print("This code demonstrates an exponential example in Section 5 in Turner and Van Zandt (2012) JMP 56, 69, with some modifications.")
    print("*******************************************")

    #preparing data
    n=500
    lambda_true=0.1
    Yobs=random.exponential(1.0/lambda_true,n)

    # start ABCpmc 
    abc=ABCpmc()
    abc.maxtryx=100000#debug magic
    abc.npart=512#debug magic

    # input model/prior
    abc.nmodel=1
    abc.model=\
    """
    /* the exponential distribution model generator */

    __device__ float model(float* par, float* xmodel, curandState* s){
    
    xmodel[0] = -log(curand_uniform(s))/par[0];

    }
    """
    abc.prior=\
    """
    #include "gengamma.h"

    __device__ void prior(float* parprior,float* xprior,curandState* s){

    xprior[0] = gammaf(parprior[0],parprior[1],s);

    return;

    }
    """

    # data and the summary statistics
    abc.n = len(Yobs)
    abc.ndata = 1
    Ysum = np.sum(Yobs)
    abc.Ysm = np.array([Ysum])
    
    # prior functional form
    def fprior():
        def f(x,parprior):
            return gammafunc.pdf(x, parprior[0],scale=1.0/parprior[1])
        return f
    abc.fprior = fprior()#
    
    #set prior parameters
    abc.parprior=np.array([0.1,0.1]) #alpha, beta
    abc.epsilon_list = np.array([3.0,1.0,1.e-1,1.e-2,1.e-3,1.e-4])

    #initial run of abc pmc
    abc.run()
    abc.check()
    plt.hist(abc.x,bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)

    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    #plotting...
    plt.hist(abc.x,bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    alpha=abc.parprior[0]+abc.n
    beta=abc.parprior[1]+Ysum
    xl = np.linspace(gammafunc.ppf(0.001, alpha,scale=1.0/beta),gammafunc.ppf(0.999, alpha,scale=1.0/beta), 100)
    plt.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\pi_\mathrm{ABC}$")
    plt.legend()
    plt.savefig("abcpmc.png")
    plt.show()
