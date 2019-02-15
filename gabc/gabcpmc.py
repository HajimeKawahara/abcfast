from pycuda.compiler import SourceModule

def gabcpmc_module ():
    source_module = SourceModule("""
    #include <stdio.h>
    #include <math.h>
    #include <curand_kernel.h>
    #include "gengamma.h"
    #define MAXTRYX 10000000

    extern __shared__ float cache[]; 
    
    /* the exponential distribution model generator */
    __device__ float model(float lambdain,curandState *s){
    
    return  -log(curand_uniform(s))/lambdain;

    }

    __device__ float prior(curandState* s,float alpha_prior,float beta_prior){

    return gammaf(alpha_prior,beta_prior,s);

    }

    #include "abcpmc_init.h"
    #include "abcpmc.h"
    #include "compute_weight.h"

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module
