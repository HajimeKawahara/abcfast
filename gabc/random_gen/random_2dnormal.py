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
    __global__ void norm2d(float* x1, float* x2, float a, float b, float c){

    unsigned long seed;
    unsigned long id;
    curandState s;
    float norm;
    float e11;
    float e12;
    float e21;
    float e22;

    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);

    float rn1 = curand_normal(&s);
    float rn2 = curand_normal(&s);
    float a2 = pow(a,2);
    float b2 = pow(b,2);
    float c2 = pow(c,2);
    float fac = sqrt(a2 - 2*a*b + b2 + 4*c2)/2;
    float sqrt_lam1 = sqrt(a/2 + b/2 - fac);        
    float sqrt_lam2 = sqrt(a/2 + b/2 + fac);

    e11 = -c/(a/2 - b/2 + fac);
    norm = sqrt(pow(e11,2)+1.0);
    e11 = sqrt_lam1*e11/norm;
    e12 = sqrt_lam1/norm;

    e21 = -c/(a/2 - b/2 - fac);
    norm = sqrt(pow(e21,2)+1.0);
    e21 = sqrt_lam2*e21/norm;
    e22 = sqrt_lam2/norm;

    x1[id] = e11*rn1 + e21*rn2;
    x2[id] = e12*rn1 + e22*rn2;

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

    source_module=gabcrm_module()
    pkernel=source_module.get_function("norm2d")
    pkernel(dev_x1,dev_x2,np.float32(a),np.float32(b),np.float32(c),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
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


    
