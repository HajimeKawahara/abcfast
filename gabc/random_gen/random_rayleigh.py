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

    #define MAXTRY 10000
    __global__ void rayleighgen(float *x){
    
    unsigned long seed;
    unsigned long id;
    curandState s;
    float a;
    float b;
    
    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);
    a = curand_normal(&s);
    b = curand_normal(&s);    
    x[id] =  sqrt(pow(a,2)+pow(b,2));

    }


    }
    

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import rayleigh as rayleighfunc
    
    print("****************************************************************")
    print("Rayleigh function (sigma=1) Random Sampler using curand_kernel.h")
    print("****************************************************************")

    alpha=2.0
    beta=3.0
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
    pkernel=source_module.get_function("rayleighgen")
    pkernel(dev_x,np.float32(alpha),np.float32(beta),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    plt.hist(x,bins=100,density=True)
#    plt.hist(np.log10(x[x>0]),bins=100,density=True)
    #plt.yscale("log")
#    plt.xscale("log")

    xl = np.linspace(rayleighfunc.ppf(0.001),rayleighfunc.ppf(0.999),1000)
    plt.plot(xl, rayleighfunc.pdf(xl))
#    plt.axvline(np.log10(np.mean(x)),color="red")
    plt.axvline(np.mean(x),color="red")
#    plt.yscale("log")
    print("mean=",np.mean(x))
    plt.title("Rayleigh Distribution")
    plt.show()
