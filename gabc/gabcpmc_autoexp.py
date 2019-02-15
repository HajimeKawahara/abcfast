import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
from utils.statutils import *


def gabcrm_module ():
    source_module = SourceModule("""
    #include <stdio.h>
    #include <math.h>
    #include <curand_kernel.h>
    #include "gengamma.h"
    #define MAXTRYX 10000000

    extern __shared__ float cache[]; 
    
    /* the exponential distribution model generator */
    __device__ float model(float lambdain,curandState *s){
    
    return  -log(curand_uniform(s))/lambdain;

    }

    __device__ float prior(curandState* s,float alpha_prior,float beta_prior){

    return gammaf(alpha_prior,beta_prior,s);

    }

    #include "abcpmc_init.h"
    #include "abcpmc.h"
    #include "compute_weight.h"

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

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
    print("Auto determination of epsilon")
    print("*******************************************")
    nthread_use_max=512 # maximun number of the threads in a block for use

    n=512 # number of the samples the should be 2**n because of thread coorporating add.
    npart=512*4 # number of the particles: should be 2**n because of thread coorporating add.
    checkpower2(n)
    checkpower2(npart)

    wide=10.0 # widespread factor of the gaussian transition kernel. Do not set less than 2 (=TVZ12).
    quantile_crit=0.75 #quantile for epsilon 
    Nmaxt=20 # max number for the T loop
    
    lambda_true=0.1
    alpha_prior=0.1
    beta_prior=0.1
    
    Yobs=random.exponential(1.0/lambda_true,n)
    Ysum=np.sum(Yobs)

    epsilon = 3.0

    seed=-1
    
    #particles
    x=np.zeros(npart)
    x=x.astype(np.float32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)

    #check trial number
    ntry=np.zeros(npart)
    ntry=ntry.astype(np.int32)
    dev_ntry = cuda.mem_alloc(ntry.nbytes)
    cuda.memcpy_htod(dev_ntry,ntry)

    #
    dist=np.zeros(npart)
    dist=dist.astype(np.float32)
    dev_dist = cuda.mem_alloc(dist.nbytes)
    cuda.memcpy_htod(dev_dist,dist)

    
    source_module=gabcrm_module()
    pkernel_init=source_module.get_function("abcpmc_init")

    #initial run
    ## n-thread, N(=nt)-block
    sharedsize=(n+1)*4 #byte
    pkernel_init(dev_x,np.float32(Ysum),np.float32(epsilon),np.int32(seed),np.float32(alpha_prior),np.float32(beta_prior),dev_dist,dev_ntry,block=(int(n),1,1), grid=(int(npart),1),shared=sharedsize)

    cuda.memcpy_dtoh(x, dev_x)
    cuda.memcpy_dtoh(dist, dev_dist)
    cuda.memcpy_dtoh(ntry, dev_ntry)
    
    epsilon=np.quantile(dist, quantile_crit)
    print("epsilon=",epsilon)
#    sys.exit("--")

    
    FR=len(x[x<0])/len(x)
    print("Fail Rate=",FR)
    if FR>0:
        print("ERROR: Increase epsilon or MAXVALX in kernel.")
        sys.exit("")
    print("mean, max, min of #try:",np.mean(ntry),np.max(ntry),np.min(ntry))
    tend=time.time()
    print("t=",tend-tstart)
    
    #========================================================================
    plt.hist(x,bins=50,label="$\epsilon$="+str(epsilon),density=True,alpha=0.3)
    plt.xlabel("lambda")
    alpha=alpha_prior+n
    beta=beta_prior+Ysum
    xl = np.linspace(gammafunc.ppf(0.001, alpha,scale=1.0/beta),gammafunc.ppf(0.999, alpha,scale=1.0/beta), 100)
    plt.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
    plt.legend()
    #========================================================================

    #window
    w=np.ones(npart)
    w=w/np.sum(w)
    w=w.astype(np.float32)
    dev_w = cuda.mem_alloc(w.nbytes)
    cuda.memcpy_htod(dev_w,w)

    Ki,Li,Ui=genalias_init(w)
    dev_Ki = cuda.mem_alloc(Ki.nbytes)
    dev_Li = cuda.mem_alloc(Li.nbytes)
    dev_Ui = cuda.mem_alloc(Ui.nbytes)

    cuda.memcpy_htod(dev_Ki,Ki)
    cuda.memcpy_htod(dev_Li,Li)
    cuda.memcpy_htod(dev_Ui,Ui)
    
    #particles (new)
    xx=np.zeros(npart)
    xx=xx.astype(np.float32)
    dev_xx = cuda.mem_alloc(xx.nbytes)
    cuda.memcpy_htod(dev_xx,xx)
    
    #weight (new)
    ww=np.zeros(npart)
    ww=ww.astype(np.float32)
    dev_ww = cuda.mem_alloc(ww.nbytes)
    cuda.memcpy_htod(dev_ww,ww)

    pkernel=source_module.get_function("abcpmc")
    wkernel=source_module.get_function("compute_weight")

    #pmc sequence
    for j in range(0,Nmaxt):
        tstartx=time.time()

        sigmat_prev = np.sqrt(wide*np.var(x))
        
        sharedsize=(n+1)*4 #byte
        pkernel(dev_xx,dev_x,np.float32(Ysum),np.float32(epsilon),dev_Ki,dev_Li,dev_Ui,np.float32(sigmat_prev),np.int32(seed),dev_dist,dev_ntry,block=(int(n),1,1), grid=(int(npart),1),shared=sharedsize)
        
        cuda.memcpy_dtoh(ntry, dev_ntry)
        print("mean, max, min of #try:",np.mean(ntry),np.max(ntry),np.min(ntry))
        cuda.memcpy_dtoh(x, dev_xx)
        #x=x
        FR=len(x[x<0])/len(x)
        print("Fail Rate=",FR)
        if FR>0:
            print("ERROR: Increase epsilon or MAXVALX in kernel.")
            sys.exit("")
        
        tend=time.time()
        print("t=",tend-tstartx)
        #update weight
        sharedsize=int(npart*4) #byte
        nthread=min(npart,nthread_use_max)
        
        wkernel(dev_ww, dev_w, dev_xx, dev_x, np.float32(sigmat_prev), block=(int(nthread),1,1), grid=(int(npart),1),shared=sharedsize)

        cuda.memcpy_dtoh(w, dev_ww)
        cuda.memcpy_dtoh(dist, dev_dist)

        if j==Nmaxt-1:
            plt.hist(x,bins=50,label="$\epsilon$="+str(epsilon),density=True,alpha=0.5)

        #UPDATE
        gampri=gammafunc.ppf(x, alpha_prior,scale=1.0/beta_prior)
        w=gampri/w
        w=w/np.sum(w)
        w=w.astype(np.float32)
        #swap
        Ki,Li,Ui=genalias_init(w)
        cuda.memcpy_htod(dev_Ki,Ki)
        cuda.memcpy_htod(dev_Li,Li)
        cuda.memcpy_htod(dev_Ui,Ui)

        dev_x, dev_xx = dev_xx, dev_x
        dev_w, dev_ww = dev_ww, dev_w

        epsilon=np.quantile(dist, quantile_crit)
        print("epsilon=",epsilon)
        

        
    print("total=",tend-tstart)

    plt.legend()    
    plt.savefig("pmc_exp.png")
    plt.show()
    #========================================================================

