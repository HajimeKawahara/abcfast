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
    __global__ void aliasgen(int* x, int* Ki, int* Li, float* Ui, int nt){

    unsigned long seed;
    unsigned long id;
    curandState s;
    float xuni;

    seed=-1;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);
    xuni = 1.0 - curand_uniform(&s);

    float pb = xuni * float(nt);
    int index = __float2int_rd(pb);

    if(xuni == 1.0){
    index = 0;
    }
    
    if(Ui[index] < pb - index){
    x[id] = Ki[index];
    }else{
    x[id] = Li[index];
    }

    }
    }


    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

def alias_init(parrs):
    parr=np.array(parrs, np.float32)
    parr = parr/np.sum(parr)

    Ui = np.ndarray(len(parrs), np.float32)
    Ki= np.zeros(len(parrs), dtype=np.int32)
    Li= np.zeros(len(parrs), dtype=np.int32)

    il, ir = 0, 0
    pairs = list(zip(parr, range(len(parrs))))
    pairs.sort()
    for parr, i in pairs:
        p = parr * len(parrs)
        while p > 1 and ir < len(Ui):
            Ki[ir] = i
            p -= 1.0 - Ui[ir]
            ir += 1
        Ui[il] = p
        Li[il] = i
        il += 1
    for i in range(ir, len(parrs)):
        Ki[i] = 0

    Ki=Ki.astype(np.int32)
    dev_Ki = cuda.mem_alloc(Ki.nbytes)
    cuda.memcpy_htod(dev_Ki,Ki)

    Li=Li.astype(np.int32)
    dev_Li = cuda.mem_alloc(Li.nbytes)
    cuda.memcpy_htod(dev_Li,Li)

    Ui=Ui.astype(np.float32)
    dev_Ui = cuda.mem_alloc(Ui.nbytes)
    cuda.memcpy_htod(dev_Ui,Ui)


    return dev_Ki,dev_Li,dev_Ui

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("***********************************************")
    print("Discrete Number Random Sampler (alias method) ")
    print("***********************************************")

    ################################################3
    parrs=[1,2,3,4,5,6,7]
    dev_Ki,dev_Li,dev_Ui=alias_init(parrs)
    
    nw=1
    nt=30000
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte

    x=np.zeros(nb)
    x=x.astype(np.int32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)

    source_module=gabcrm_module()
    pkernel=source_module.get_function("aliasgen")
    pkernel(dev_x,dev_Ki,dev_Li,dev_Ui,np.int32(len(parrs)),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    plt.hist(x,bins=100)
    plt.plot(range(0,len(parrs)),np.array(parrs)*len(x)/np.sum(parrs),"*",label="$p_i$")
    plt.errorbar(range(0,len(parrs)),np.array(parrs)*len(x)/np.sum(parrs),yerr=np.sqrt(np.array(parrs)*len(x)/np.sum(parrs)),fmt=".")

    plt.legend()
    plt.xticks(range(0,len(parrs)))
    plt.show()
