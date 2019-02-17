from gabc.gabcpmc import *
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
from gabc.utils.statutils import *

def checkpower2(n):
    logn=np.log2(n)
    if logn - int(logn) > 0.0:
        print("n=",n)
        sys.exit("Use 2^(integer) as n.")
    return
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from scipy.stats import expon as expfunc
    from scipy.stats import gamma as gammafunc
    import time
    import sys
    
    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC PMC Method.")
    print("This code demonstrates an exponential example in Section 5 in Turner and Van Zandt (2012) JMP 56, 69, with some modifications.")
    print("*******************************************")

    #data
    n=500
    lambda_true=0.1
    Yobs=random.exponential(1.0/lambda_true,n)
    #Ysum=np.sum(Yobs)

    #start ABCpmc 
    abc=ABCpmc(512*8,Yobs)
    
    abc.model=\
    """
    /* the exponential distribution model generator */
    __device__ float model(float lambdain,curandState *s){
    
    return  -log(curand_uniform(s))/lambdain;

    }
    """
    # set random number generator of a prior
    abc.prior=\
    """
    __device__ float prior(curandState* s,float* parprior){

    return gammaf(parprior[0],parprior[1],s);

    }
    """
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
    plt.hist(abc.x,bins=50,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)

    #pmc sequence
    for j,epsilon in enumerate(abc.epsilon_list[1:]):
        abc.run()
        abc.check()

    #plotting...
    plt.hist(abc.x,bins=50,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    alpha=abc.parprior[0]+abc.n
    beta=abc.parprior[1]+np.sum(abc.Yobs)
    xl = np.linspace(gammafunc.ppf(0.001, alpha,scale=1.0/beta),gammafunc.ppf(0.999, alpha,scale=1.0/beta), 100)
    plt.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\pi_\mathrm{ABC}$")

    plt.legend()
    plt.savefig("abcpmc.png")
    plt.show()