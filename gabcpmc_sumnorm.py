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
    print("GPU ABC PMC Method.")
    print("This code demonstrates a normal+normal distribution. Beaumont+2009")
    print("*******************************************")

    #preparing data
    nsample=500
    Yobs=0.0

    # start ABCpmc 
    abc=ABCpmc()
    abc.wide=2.0
    abc.Ecrit=0.0
    abc.maxtryx=100000#debug magic
    abc.npart=512*16#debug magic

    # input model/prior
    abc.nparam=1
    abc.model=\
    """
    /* the double normal distribution model generator */
    #include "gennorm.h"

    __device__ float model(float* Ysim, float* param, curandState* s, float* aux, int isample){
    
    float cl=curand_uniform(s);
    int i=int(cl*2.0);
    float siga[2];
    siga[0]=1.0;
    siga[1]=1.e-1;

    Ysim[0] =  normf(param[0],siga[i], s);

    }
    """
    
    # prior 
    def fprior():
        def f(x):
            mask=(x<10.0)*(x>-10.0)
            arr=np.zeros(len(x))
            arr[mask]=1.0
            return arr
        return f
    abc.fprior = fprior()#

    abc.prior=\
    """
    #include <curand_kernel.h>

    __device__ void prior(float* param,curandState* s){

    param[0] = (curand_uniform(s)-0.5)*20.0;

    return;

    }
    """

    # data and the summary statistics
    abc.nsample = 1
    abc.ndata = 1
    Ysum = Yobs
    abc.Ysm = np.array([Ysum])
    
    
    #set prior parameters
    abc.epsilon_list = np.array([2.0,1.5,1.0,0.5,1.e-2])

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
    
    xref=np.linspace(-3.0,3.0,1000)
    
    print(tend-tstart,"sec")
    print(abc.xres())
    
    #plotting...
    fig=plt.figure()
    ax=fig.add_subplot(211)
    ax.hist(abc.x,bins=200,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.3)

    ax.hist(abc.xres(),bins=200,label="resampled",density=True,alpha=0.1)
    ax.plot(xref,0.5*normfunc.pdf(xref,0.0,1.0)+0.5*normfunc.pdf(xref,0.0,1.e-1))
    ax.legend()
    ax=fig.add_subplot(212)
    ax.plot(abc.x,abc.w,".")
    plt.ylabel("weight")
    plt.xlim(-3,3)
    plt.ylim(0,np.max(abc.w))
    plt.savefig("sumnorm.png")
    plt.show()
