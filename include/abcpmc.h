/* abcpmc */
/* i=0,...,n-1, are X[i], i=n,...,n+npart-1, are previous X[i] i=n+npart is used for a block prior (xast) */
#include "genalias.h"

extern "C"{

  __global__ void abcpmc(float* x, float* xprev, float Ysum, float epsilon, int* Ki, int* Li, float* Ui, float sigmat_prev, int seed, float* dist, int* ntry, int ptwo){

    curandState s;
    int cnt = 0;
    float p;
    float xast;
    float xastast;
    float rho;
    int n = blockDim.x;
    int npart = gridDim.x;
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
    isel=aliasgen(Ki, Li, Ui, npart,&s);
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


    /* SUMMARY STATISTICS (currently summation with power of 2 samples) */
    /* thread cooperating computation of rho */        
    /*    int i = n/2; */
    int i = ptwo;
    while(i !=0) {
        if (ithread + i < n && ithread < i){
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
    dist[iblock]=rho;
    }

    return;
    }

    __syncthreads();
    /* ===================================================== */
    

    }
    }
    }
