#include <curand_kernel.h>
#include "genbinomial.h"

extern "C"{
    __global__ void abcrm(float *x, int Yobs, int n, int epsilon){

    unsigned long seed;
    unsigned long id;
    curandState s;
    int X;
    int cnt = 0;
    float p;

    seed=10;
    id = blockIdx.x;
    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRY){
    printf("EXCEED MAXTRY \\n");
    x[id] = 0.0;
    return;
    }

    /* sample p from the uniform distribution */
    p = curand_uniform(&s);

    /* sampler */
    X = binomialf(n,p,&s);

    if(rho(X,Yobs)<epsilon){
    x[id] = p;
    return;
    }

    }


    }

    }
