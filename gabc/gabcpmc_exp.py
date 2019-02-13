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
    #include "genalias.h"

    #define MAXTRYX 10000000

    /* stucture of shared memory */
    /* abcpmc_init */
    /* i=0,...,n-1, are X[i], i=n is used for a block prior (xast) */
    
    /* abcpmc */
    /* i=0,...,n-1, are X[i], i=n,...,n+nt-1, are previous X[i] i=n+nt is used for a block prior (xast) */


    extern __shared__ float cache[]; 

    /* the exponential distribution model generator */
    __device__ float model(float lambdain,curandState *s){
    
    return  -log(curand_uniform(s))/lambdain;

    }


    __device__ float prior(curandState* s,float alpha_prior,float beta_prior){

    return gammaf(alpha_prior,beta_prior,s);

    }


    extern "C"{
    __global__ void abcpmc_init(float* x, float Ysum, float epsilon, int seed, float alpha_prior,float beta_prior, int* ntry){

    curandState s;
    int cnt = 0;
    float p;
    float xast;
    float rho;
    int n = blockDim.x;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*n + ithread;
    
    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRYX){
    if(ithread==0){
    printf("EXCEED MAXTRYX. iblock=%d \\n",iblock);
    x[iblock] = -1.0;
    ntry[iblock]=MAXTRYX;

    }
    return;
    }


    /* sampling a prior from the Gamma distribution */
    if(ithread == 0){
    cache[n] = prior(&s,alpha_prior,beta_prior);
    }
    __syncthreads();
    /* ===================================================== */
    
    xast = cache[n];


    /* sample p from the uniform distribution */

    cache[ithread] = model(xast,&s);

    __syncthreads();
    /* ===================================================== */



    /* thread cooperating computation of rho */        
    int i = n/2;
    while(i !=0) {
        if (ithread < i){
        cache[ithread] += cache[ithread + i];
        }
        __syncthreads();
        i /= 2;
    }

  __syncthreads();
    /* ===================================================== */

    rho = abs(cache[0] - Ysum)/n;

    if(rho<epsilon){

    if(ithread==0){
    x[iblock] = xast;
    ntry[iblock]=cnt;
    }

    return;
    }

    __syncthreads();
    /* ===================================================== */
    

    }
    }

    __global__ void abcpmc(float* x, float* xprev, float Ysum, float epsilon, int* Ki, int* Li, float* Ui, float sigmat_prev, int seed, int* ntry){

    curandState s;
    int cnt = 0;
    float p;
    float xast;
    float xastast;
    float rho;
    int n = blockDim.x;
    int nt = gridDim.x;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*n + ithread;
    float uni;
    int isel;
    

    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRYX){
    if(ithread==0){
    printf("EXCEED MAXTRYX. iblock=%d \\n",iblock);
    x[iblock] = -1.0;
    ntry[iblock]=MAXTRYX;

    }
    return;
    }


    /* sampling a prior from the previous posterior*/
    if(ithread == 0){
    isel=aliasgen(Ki, Li, Ui, nt,&s);
    xast = xprev[isel];
    xastast = xast + curand_normal(&s)*sigmat_prev;

    cache[n] = xastast;
    }
    __syncthreads();
    /* ===================================================== */
    
    xastast = cache[n];

    /* sample p from the uniform distribution */

    cache[ithread] = model(xastast,&s);

    __syncthreads();
    /* ===================================================== */



    /* thread cooperating computation of rho */        
    int i = n/2;
    while(i !=0) {
        if (ithread < i){
        cache[ithread] += cache[ithread + i];
        }
        __syncthreads();
        i /= 2;
    }

  __syncthreads();
    /* ===================================================== */

    rho = abs(cache[0] - Ysum)/n;

    if(rho<epsilon){

    if(ithread==0){
    x[iblock] = xastast;
    ntry[iblock]=cnt;
    }

    return;
    }

    __syncthreads();
    /* ===================================================== */
    

    }
    }

    __global__ void compute_weight(float* wnew, float* wprev, float* xnew, float* xprev, float sigmat_prev){
    int n = blockDim.x;
    int nt = gridDim.x;

    /* prev:=t-1, new:=t in TVZ12 */ 
    /* iblock := i in TVZ12 */
    int iblock = blockIdx.x;

    /* ithread := j in TVZ12 */
    int ithread = threadIdx.x;
    float qf;

    /* computing qf (Gaussian transition kernel) */
    qf = exp(-pow((xprev[ithread] - xnew[iblock]),2)/(pow(sigmat_prev,2)));

    /* thread cooperating computation of a denominater */        
    cache[ithread] = wprev[ithread]*qf;
    __syncthreads();

    int i = n/2;
    while(i !=0) {
        if (ithread < i){
        cache[ithread] += cache[ithread + i];
        }
        __syncthreads();
        i /= 2;
    }

    __syncthreads();
    /* ===================================================== */

    if (ithread==0){
    /* computing weight */
    wnew[iblock] = cache[0];

    return;

    }else{

    return;

    }


    }


    }

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module

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
    print("This code demonstrates an exponential example in Section 5 in Turner and Van Zandt (2012) JMP 56, 69")
    print("*******************************************")
    n=512 #should be 2**n because of thread coorporating add.
    
    lambda_true=0.1
    alpha_prior=0.1
    beta_prior=0.1
    
    Yobs=random.exponential(1.0/lambda_true,n)
    Ysum=np.sum(Yobs)

#    plt.hist(Yobs,bins=100,density=True,alpha=0.5,label="numpy random")
#    xl = np.linspace(expfunc.ppf(0.001, scale=1.0/lambda_true),expfunc.ppf(0.999, scale=1.0/lambda_true), 100)
#    plt.plot(xl, expfunc.pdf(xl, scale=1.0/lambda_true))
#    plt.axvline(Ysum/n)
#    plt.title("exponential distribution, lambda="+str(lambda_true))
#    plt.show()

    epsilon_list = np.array([3.0,1.0,1.e-1,1.e-3,1.e-4,1.e-5])
#    epsilon_list = np.array([3.0,1.0,1.e-1,1.e-2,1.e-3,1.e-4,1.e-5])

    seed_list=[3,76,81,39,-34,23,83,12]
    Niter=len(epsilon_list)

    nt=2048 #should be 2**n because of thread coorporating add.
    sharedsize=0 #byte

    #data
    #Yobs=Yobs.astype(np.float32)
    #    dev_y = cuda.mem_alloc(Yobs.nbytes)
    #    cuda.memcpy_htod(dev_y,Yobs)
    
    #particles
    x=np.zeros(nt)
    x=x.astype(np.float32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)

    #check trial number
    ntry=np.zeros(nt)
    ntry=ntry.astype(np.int32)
    dev_ntry = cuda.mem_alloc(ntry.nbytes)
    cuda.memcpy_htod(dev_ntry,ntry)

    
    source_module=gabcrm_module()
    pkernel_init=source_module.get_function("abcpmc_init")

    #initial run

    # n-thread, N(=nt)-block
    sharedsize=(n+1)*4 #byte
    seed=seed_list[0]
    epsilon=epsilon_list[0]
    pkernel_init(dev_x,np.float32(Ysum),np.float32(epsilon),np.int32(seed),np.float32(alpha_prior),np.float32(beta_prior),dev_ntry,block=(int(n),1,1), grid=(int(nt),1),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)
    
    cuda.memcpy_dtoh(ntry, dev_ntry)

    #x=x
    FR=len(x[x<0])/len(x)
    print("Fail Rate=",FR)
    if FR>0:
        print("ERROR: Increase epsilon or MAXVALX in kernel.")
        sys.exit("")
    print("mean, max, min of #try:",np.mean(ntry),np.max(ntry),np.min(ntry))
    tend=time.time()
    print("t=",tend-tstart)
    

    #plt.plot(np.log10(x),".")
    #plt.show()

    plt.hist(x,bins=30,label="$\epsilon$="+str(epsilon),density=True,alpha=0.3)
    plt.xlabel("lambda")
    alpha=alpha_prior+n
    beta=beta_prior+Ysum
    xl = np.linspace(gammafunc.ppf(0.001, alpha,scale=1.0/beta),gammafunc.ppf(0.999, alpha,scale=1.0/beta), 100)
    plt.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
    plt.legend()
    #========================================================================

    #window
    w=np.ones(nt)
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
    xx=np.zeros(nt)
    xx=xx.astype(np.float32)
    dev_xx = cuda.mem_alloc(xx.nbytes)
    cuda.memcpy_htod(dev_xx,xx)
    
    #weight (new)
    ww=np.zeros(nt)
    ww=ww.astype(np.float32)
    dev_ww = cuda.mem_alloc(ww.nbytes)
    cuda.memcpy_htod(dev_ww,ww)

    pkernel=source_module.get_function("abcpmc")
    wkernel=source_module.get_function("compute_weight")

    #pmc sequence
    for j,epsilon in enumerate(epsilon_list[1:]):
        tstartx=time.time()
        print(j,epsilon)
        seed = seed_list[j+1]
        sigmat_prev = np.sqrt(2.0*np.var(x))
        sharedsize=(n+1)*4 #byte
        pkernel(dev_xx,dev_x,np.float32(Ysum),np.float32(epsilon),dev_Ki,dev_Li,dev_Ui,np.float32(sigmat_prev),np.int32(seed),dev_ntry,block=(int(n),1,1), grid=(int(nt),1),shared=sharedsize)
        
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
        plt.hist(x,bins=10,label="$\epsilon$="+str(epsilon),density=True,alpha=0.3)
        
        #update weight
        sharedsize=(nt)*4 #byte
#    __global__ void compute_weight(float* wnew, float* wprev, float* xnew, float* xprev, float sigmat_prev){

        wkernel(dev_ww, dev_w, dev_xx, dev_x, np.float32(sigmat_prev), block=(int(nt),1,1), grid=(int(nt),1),shared=sharedsize)

        cuda.memcpy_dtoh(w, dev_ww)
        
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

    print("total=",tend-tstart)

    plt.legend()    
    plt.savefig("pmc_exp.png")
    plt.show()
    #========================================================================

