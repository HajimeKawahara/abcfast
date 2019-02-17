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
    #define MAXTRY 10000

    extern "C"{

    __global__ void gammagen(float *x, float ain, float bin){

    unsigned long seed;
    unsigned long id;
    curandState s;
    unsigned long cnt=0;
    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);
    float d, c, y, v, u, a;

    if(ain > 1.0){
    a = ain;
    }else if(ain > 0.0){
    a = ain + 1.0;
    }else{
    return;
    }

    d = a - 1. / 3.;
    c = 1. / sqrt(9. * d);
    for (;;) {
    do {
    y =curand_normal(&s);
    v = 1. + c * y;
    } while (v <= 0.);

    cnt=cnt+1;
    if(cnt > MAXTRY){
    printf("EXCEED MAXTRY");
    return;
    }

    v = v * v * v;
    u = curand_uniform(&s);
    if (u < 1. - 0.0331 * (y * y) * (y * y)) {

    if(a > ain){
    x[id] = (d * v)*pow(curand_uniform(&s),1.0/ain)/bin;
    return;
    }else{
    x[id] = (d * v)/bin;
    return;
    }

    }
    if (log(u) < 0.5 * y * y + d * (1. - v + log(v))) {

    if(a > ain){
    x[id] = (d * v)*pow(curand_uniform(&s),1.0/ain)/bin;
    return;
    }else{
    x[id] = (d * v)/bin;
    return;
    }

    }

    }

    }
    }
    

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma as gammafunc
    
    print("********************************************")
    print("Gamma function Random Sampler using curand_kernel.h")
    print("by Marsaglia and Tsang’s method")
    print(":Gamma[alpha,beta] for alpha > 0, beta >0")
    print("********************************************")

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
    pkernel=source_module.get_function("gammagen")
    pkernel(dev_x,np.float32(alpha),np.float32(beta),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    plt.hist(x,bins=100,density=True)
#    plt.hist(np.log10(x[x>0]),bins=100,density=True)
    #plt.yscale("log")
#    plt.xscale("log")

    xl = np.linspace(gammafunc.ppf(0.001, alpha,scale=1.0/beta),gammafunc.ppf(0.999, alpha,scale=1.0/beta), 100)
    plt.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta))
#    plt.axvline(np.log10(np.mean(x)),color="red")
    plt.axvline(np.mean(x),color="red")
    plt.yscale("log")
    print("mean=",np.mean(x))
    plt.title("Gamma Distribution, alpha="+str(alpha)+" beta="+str(beta))
    plt.show()
