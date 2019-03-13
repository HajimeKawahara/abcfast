from gabc.gabcpmc import *
from gabc.utils.statutils import *
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from scipy.stats import gamma as gammafunc
    from scipy.stats import norm as normfunc
    from scipy.special import gamma as gammaform

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
    abc.maxtryx=100000000#debug magic
    abc.npart=512#debug magic

    # input model/prior
    abc.nparam=1
    abc.nsubject = nsub
    abc.nsample = nsample
#    abc.Ecrit=0.0 #resampling off

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
    logitp = normf(hparam[0],exp(hparam[1]),s);
    
    /* exp(18)(1+exp(18)) = 1.00000 effectively */
    el = min(18.0,logitp);

    el = exp(el);
    param[0] = el/(1.0 + el);
    
    return;

    }
    """

    mumu=0.0
    ximu=100.0
    alphas=0.1
    betas=0.1
    # hyperprior functional form (PDF) hyperparameter = (mu,log(sigma))
    def fhprior():
        def f(x):
            h0=normfunc.pdf(x[:,0],loc=mumu,scale=np.sqrt(ximu) )
            sigma=np.exp(x[:,1])
            h1=(betas**alphas)/gammaform(alphas)*((1.0/sigma)**(alphas+1.0))*np.exp(-betas/sigma)
            return h0*h1
        return f
    abc.fhprior = fhprior()#

    abc.hyperprior=\
    """
    #include "gengamma.h"

    __device__ void hyperprior(float* hparam,curandState* s){

    hparam[0] = normf(0.0,10.0,s);
    hparam[1] = log(1.0/gammaf(0.1,0.1,s));

    return;

    }
    """
        
    # data and the summary statistics
    abc.ndata = 1
    abc.nhparam = 2 
    abc.Ysm = Ysum_obs
    
    abc.wide=2.0
    #set prior parameters
    abc.epsilon_list = np.array([1.0,0.5,0.3,0.1,0.07,0.06,0.05,0.03,0.02])
    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()
        
    #plot 0
    xw0=np.copy(abc.xw)


    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    tend = time.time()
    print(tend-tstart,"sec")

    xw0=np.copy(abc.xw)


    fig=plt.figure(figsize=(10,5))

    ax=fig.add_subplot(121)    
    #    ax.hist(abc.z,bins=100,alpha=0.3,color="gray")    
    for isub in range(0,abc.nsubject):
        ax.hist(abc.zw[:,isub],bins=30,alpha=0.3,label="subject #"+str(isub),color="C"+str(isub))
    plt.xlabel("p")
    plt.legend()
    for j,ip in enumerate(ptrue):
        plt.axvline(ip,ls="dashed",color="C"+str(j))
    ax=fig.add_subplot(122)
    cl=ax.scatter(abc.xw[:,0],abc.xw[:,1],c=abc.w,alpha=0.2)
    plt.colorbar(cl)
    plt.xlim(-3,2)
    plt.ylim(-3.5,2)
    plt.axhline(np.log(0.5))
    plt.axvline(-1.0)
    plt.xlabel("$\mu$")
    plt.ylabel("log $\sigma$")
    plt.savefig("hibin.png")
    #plot Np-1
    plt.show()

