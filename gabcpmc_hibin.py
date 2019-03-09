from gabc.gabcpmc import *
from gabc.utils.statutils import *
        
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
    for j,p in enumerate(ptrue):
        Ysum_obs=np.concatenate([Ysum_obs,[random.binomial(nres,p)]])
    print("data:",Ysum_obs)
    # start ABCpmc 
    abc=ABCpmc(hyper=True)
    abc.maxtryx=100#debug magic
    abc.npart=512#debug magic

    # input model/prior
    abc.nparam=1
    abc.nsubject = nsub
    abc.nsample = nsample

    abc.model=\
    """
    /* the exponential distribution model generator */

    __device__ void model(float* Ysim, float* param, curandState* s){
    
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

    __device__ void prior(float* param,float* hparam,curandState* s){

    float logitp;
    float el;
    logitp = normf(hparam[0],hparam[1],s);
    /* exp(18)(1+exp(18)) = 1.00000 effectively */
    el = min(18.0,logitp);
    el = exp(el);
    param[0] = el/(1.0 + el);
    
    return;

    }
    """

    mumu=0.0
    ximu=10000.0
    alphas=0.1
    betas=0.1
    # prior functional form
    def fhprior():
        def f(x):
            h0=normfunc.pdf(x,loc=mumu,scale=np.sqrt(ximu) )
            h1=1.0/gammafunc.pdf(x, alphas,scale=1.0/betas)
            return np.array([h0,h1])
        return f
    abc.fhprior = fhprior()#

    abc.hyperprior=\
    """
    #include "gengamma.h"

    __device__ void hyperprior(float* hparam,curandState* s){

    hparam[0] = normf(0.0,10.0,s);
    hparam[1] = 1.0/gammaf(0.1,0.1,s);

    return;

    }
    """
        
    # data and the summary statistics
    abc.ndata = 1
    abc.nhparam = 2 
    abc.Ysm = Ysum_obs
    
    
    #set prior parameters
    abc.epsilon_list = np.array([3.0,1.0,1.e-1,1.e-3,1.e-4,1.e-5])

    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()
    xw0=np.copy(abc.xw)
    print(np.shape(xw0))
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.hist(xw0[:,0],bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    ax=fig.add_subplot(122)
    ax.hist(np.log(xw0[:,1]),bins=20,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    plt.show()
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
