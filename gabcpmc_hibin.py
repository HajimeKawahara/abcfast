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
    print("GPU ABC PMC Method for Hierarchical Binomial Distribution.")
    print("This code demonstrates an exponential example in Section 6 in Turner and Van Zandt (2012) JMP 56, 69, with some modifications.")
    print("*******************************************")

    #preparing data

    nsub = 4 #number of subjects    
    nres= 100 # number of observation per subject
    nsample=nsub*nres

    logitp = random.normal(loc=-1.0,scale=np.sqrt(0.5),size=nsub)
    ptrue = np.exp(logitp)/(1 + np.exp(logitp))
    Ysum_obs=np.array([])
    subindex=np.array([],dtype=np.int32)
    for j,p in enumerate(ptrue):
        Ysum_obs=np.concatenate([Ysum_obs,[random.binomial(nres,p)]])
        subindex=np.concatenate([subindex,np.ones(nres,dtype=np.int32)*j])
    
    # start ABCpmc 
    abc=ABCpmc(hyper=True)
    abc.maxtryx=10000000#debug magic
    abc.npart=512#debug magic

    # input model/prior
    abc.nparam=1
    abc.model=\
    """
    /* the exponential distribution model generator */

    __device__ float model(float* param, float* Ysim, curandState* s){
    
    if (curand_uniform(s) <= param[0]){
    Ysim[0] = 1.0;
    }else{
    Ysim[0] = 0.0;
    }

    }
    """
    abc.prior=\
    """
    #include "gennorm.h"

    __device__ void prior(float* hparam,float* param,curandState* s){

    float logitp;

    logitp = normf(hparam[0],hparam[1],s);
    param[0] = exp(logitp)/(1.0 + exp(logitp));

    return;

    }
    """

    abc.parhyper=np.array([0.0,100.0,0.1,0.1]) #mu_mu, xi_mu, alpha_sigma, beta_sigma
    abc.hyperprior=\
    """
    #include "gengamma.h"

    __device__ void hyperprior(float* parhyper, float* hparam,curandState* s){

    hparam[0] = normf(parhyper[0],parhyper[1],s);
    hparam[1] = 1.0/gammaf(parhyper[2],parhyper[3],s);

    return;

    }
    """
    
    
    # data and the summary statistics
    abc.subindex = subindex
    abc.ndata = 1
    abc.nhparam = 2 
    abc.Ysm = Ysum_obs
    
    # prior functional form
    def fprior():
        def f(x,hparam):
            return gammafunc.pdf(x, hparam[0],scale=1.0/hparam[1])
        return f
    abc.fprior = fprior()#
    
    #set prior parameters
    abc.epsilon_list = np.array([3.0,1.0,1.e-1,1.e-3,1.e-4,1.e-5])

    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()
    plt.hist(abc.x,bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    sys.exit()
    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    tend = time.time()
    print(tend-tstart,"sec")
    
    #plotting...
    plt.hist(abc.x,bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    alpha=abc.hparam[0]+abc.n
    beta=abc.hparam[1]+Ysum
    xl = np.linspace(gammafunc.ppf(0.001, alpha,scale=1.0/beta),gammafunc.ppf(0.999, alpha,scale=1.0/beta), 100)
    plt.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\pi_\mathrm{ABC}$")
    plt.legend()
    plt.savefig("abcpmc.png")
    plt.show()
