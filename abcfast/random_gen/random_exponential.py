import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

def gabcrm_module ():
    source_module = SourceModule("""

    #include <curand_kernel.h>

    extern "C"{
    __global__ void expgen(float *x, float lambdain){

    unsigned long seed;
    unsigned long id;
    curandState s;
    
    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);

    x[id] = -log(curand_uniform(&s))/lambdain;

    }
    }

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import expon as expfunc
    from numpy import random
    print("********************************************")
    print("Exponential distribution Random Sampler using curand_kernel.h")
    print("********************************************")

    lambdain=0.1
    
    nw=1
    nt=100000
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte
    x=np.zeros(nb)
    x=x.astype(np.float32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)

    source_module=gabcrm_module()
    pkernel=source_module.get_function("expgen")
    pkernel(dev_x,np.float32(lambdain),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    plt.hist(x,bins=100,density=True,alpha=0.5,label="gabc exp")
    #xn=random.exponential(1.0/lambdain,nt)
    #plt.hist(xn,bins=100,density=True,alpha=0.5,label="numpy random")
    xl = np.linspace(expfunc.ppf(0.001, scale=1.0/lambdain),expfunc.ppf(0.999, scale=1.0/lambdain), 100)
    plt.plot(xl, expfunc.pdf(xl, scale=1.0/lambdain))
    #plt.axvline(np.sum(x)/nt)
    plt.title("exponential distribution, lambda="+str(lambdain))
    plt.show()
