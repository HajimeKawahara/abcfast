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
    __global__ void uniformgen(float *x){

    unsigned long seed;
    unsigned long id;
    curandState s;
    
    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);

    x[id] = curand_uniform(&s);

    }
    }

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("********************************************")
    print("Uniform Random Sampler using curand_kernel.h")
    print("********************************************")

    nw=1
    nt=10000
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte
    x=np.zeros(nb)
    x=x.astype(np.float32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)

    source_module=gabcrm_module()
    pkernel=source_module.get_function("uniformgen")
    pkernel(dev_x,block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    plt.hist(x)
    plt.show()
