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
    __global__ void bingen(float *x, float p, int n){

    unsigned long seed;
    unsigned long id;
    curandState s;
    float val;

    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);
    val = 0.0;

    for (int i = 0; i < n; i++){

    if (curand_uniform(&s) <= p){
    val++;
    }

    }

    x[id]=val;

    }
    }

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("********************************************")
    print("Binomial Random Sampler using curand_kernel.h")
    print("Bin(n,p)")
    print("********************************************")

    p=0.3
    n=5
    
    nw=1
    nt=10000
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte
    x=np.zeros(nb)
    x=x.astype(np.float32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)
    print(n)
    source_module=gabcrm_module()
    pkernel=source_module.get_function("bingen")
    pkernel(dev_x,np.float32(p),np.int32(n),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    plt.hist(x)
    plt.show()
