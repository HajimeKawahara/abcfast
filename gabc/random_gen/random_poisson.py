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
    #define NMAXPOI 1000
    #include <curand_kernel.h>

    extern "C"{
    __global__ void poigen(float *x, float p){

    unsigned long seed;
    unsigned long id;
    curandState s;
    float val;
    float c;
    int m;
    float d;
    float pu;
    float pl;
    int Xu;
    int Xl;
    float V;
    float U;
    int i;

    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);
    c = 1.0/p;
    m = int(p);
    d = - p + m*log(p);

    for (int k=0; k<m; k++){
    d = d - log(float(k+1));
    }

    d = exp(d);
    pu=d;
    pl=d;
    Xu=m;
    Xl=m;


    i=0;
    V = curand_uniform(&s) - pu;

    while(i < NMAXPOI){
    i=i+1;

    if(V <= 0.0){
    x[id]=float(Xu);
    return;
    }

    U = V;
    if(Xl > 0.0){
    pl=pl*c*float(Xl);
    Xl = Xl - 1;
    V = U - pl;

    if(V < 0.0){
    x[id]=float(Xl);
    return;
    }
    U = V;
    }

    Xu = Xu+1;
    pu = pu*p/float(Xu);
    V = U - pu;

    }

    x[id]=-1.0;
    return;
    
    }
    }

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson
    
    print("********************************************")
    print("Poisson Random Sampler using curand_kernel.h")
    print("Poisson(p)")
    print("********************************************")

    p=103.5
    
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
    pkernel=source_module.get_function("poigen")
    pkernel(dev_x,np.float32(p),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)
    print(x)
    plt.hist(x,bins=300)
    xref=range(int(np.min(x)),int(np.max(x)))
    plt.plot(xref,poisson.pmf(xref,p)*nt)
    plt.show()
