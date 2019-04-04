import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

def norm2d_module ():
    source_module = SourceModule("""

    #include <curand_kernel.h>
    #define NMODEL 2

    extern "C"{
    __global__ void norm2d(float* x1, float* x2, float* Qmat){

    unsigned long seed;
    unsigned long id;
    curandState s;
    float rn[NMODEL];

    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);
    
    for (int m=0; m<NMODEL; m++){
    rn[m] = curand_normal(&s);
    }

    x1[id]=0.0;
    x2[id]=0.0;
    int m;
    for (int k=0; k<NMODEL; k++){
    /* SHOULD CHECK */
    m=0;
/*    x1[id] += Qmat[k*NMODEL+m]*rn[k]; */
    x1[id] += Qmat[m*NMODEL+k]*rn[k]; 
    m=1;
/*    x2[id] += Qmat[k*NMODEL+m]*rn[k]; */
    x2[id] += Qmat[m*NMODEL+k]*rn[k]; 
    }


    }
    }

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

def eigen_2Dsym(a,b,c):
    #M=[[a,c],[c,b]]
    lam1=a/2 + b/2 - np.sqrt(a**2 - 2*a*b + b**2 + 4*c**2)/2        
    e1=np.array([-c/(a/2 - b/2 + np.sqrt(a**2 - 2*a*b + b**2 + 4*c**2)/2),1])
    lam2=a/2 + b/2 + np.sqrt(a**2 - 2*a*b + b**2 + 4*c**2)/2
    e2=np.array([-c/(a/2 - b/2 - np.sqrt(a**2 - 2*a*b + b**2 + 4*c**2)/2),1])
    e1=e1/np.linalg.norm(e1)
    e2=e2/np.linalg.norm(e2)
    
    return lam1,lam2,e1,e2

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    target_cov  = np.matrix([[  1.0, 0.7], [  0.7, 0.6]])
    mean = [0.0, 0.0]

    a=target_cov[0,0]
    b=target_cov[1,1]
    c=target_cov[1,0]
    
#    Q * [r1,r2]

    print("********************************************")
    print("Gaussian Random Sampler using curand_kernel.h")
    print("********************************************")

    nw=1
    nt=100000
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte


    
    x1=np.zeros(nb)
    x1=x1.astype(np.float32)
    dev_x1 = cuda.mem_alloc(x1.nbytes)
    cuda.memcpy_htod(dev_x1,x1)

    x2=np.zeros(nb)
    x2=x2.astype(np.float32)
    dev_x2 = cuda.mem_alloc(x2.nbytes)
    cuda.memcpy_htod(dev_x2,x2)

    [eigenvalues, eigenvectors] = np.linalg.eig(target_cov)
    l = np.matrix(np.diag(np.sqrt(eigenvalues)))
    Q = np.matrix(eigenvectors) * l
    Qmat=(Q.flatten()).astype(np.float32)
    dev_Qmat = cuda.mem_alloc(Qmat.nbytes)
    cuda.memcpy_htod(dev_Qmat,Qmat)

#    dev_cov=(target_cov.flatten()).astype(np.float32)
    
    source_module=norm2d_module()
    pkernel=source_module.get_function("norm2d")
    
    pkernel(dev_x1,dev_x2,dev_Qmat,block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x1, dev_x1)
    cuda.memcpy_dtoh(x2, dev_x2)

    xx=np.random.multivariate_normal(mean,target_cov,nt)
    xx1=xx[:,0]
    xx2=xx[:,1]
    print(len(x1[x1==x1]),len(x2[x2==x2]),len(xx1[xx1==xx1]),len(xx2[xx2==xx2]),)
    
    fig=plt.figure()
    ax1=fig.add_subplot(221)
    ax1.plot(x1,x2,".",alpha=0.01)
    plt.xlim(-5,5)
    plt.ylim(-5,5)    
    ax2=fig.add_subplot(222)
    ax2.plot(xx[:,0],xx[:,1],".",alpha=0.01)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    ax1=fig.add_subplot(223)
    ax1.hist(x1,bins=50,alpha=0.3,density=True,range=(-3,3))
    ax1.hist(xx[:,0],bins=50,alpha=0.3,density=True,range=(-3,3))
    #plt.yscale("log")

    ax1=fig.add_subplot(224)
    ax1.hist(x2,bins=50,alpha=0.3,density=True,range=(-3,3))
    ax1.hist(xx[:,1],bins=50,alpha=0.3,density=True,range=(-3,3))
    #plt.yscale("log")

    plt.show()


    
