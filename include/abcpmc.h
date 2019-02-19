/* abcpmc */
/* i=0,...,n-1, are X[i], i=n,...,n+npart-1, are previous X[i] i=n+npart is used for a block prior (xast) */
#include "genalias.h"

extern "C"{

  __global__ void abcpmc(float* x, float* xprev, float Ysum, float epsilon, int* Ki, int* Li, float* Ui, float sigmat_prev, int seed, float* dist, int* ntry, int ptwo){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int n = blockDim.x;
    int npart = gridDim.x;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*n + ithread;
    float uni;
    int isel;
    float xprior[NMODEL];
    
    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRYX){
      if(ithread==0){
	printf("EXCEED MAXTRYX. iblock=%d \\n",iblock);
	  for (int m=0; m<NMODEL; m++){
	    x[NMODEL*iblock + m] = -1.0;
	  }
	  ntry[iblock]=MAXTRYX;	  
      }
      return;
    }
    

    /* sampling a prior from the previous posterior*/
    if(ithread == 0){
      isel=aliasgen(Ki, Li, Ui, npart,&s);
      for (int m=0; m<NMODEL; m++){
	xprior[m] = xprev[NMODEL*isel+m] + curand_normal(&s)*sigmat_prev;
	cache[n+m] = xprior[m];
      }
    }
    __syncthreads();
    /* ===================================================== */
    for (int m=0; m<NMODEL; m++){
      xprior[m] = cache[n+m];
    }

    /* sample p from the uniform distribution */

    cache[ithread] = model(xprior,&s);

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
      for (int m=0; m<NMODEL; m++){
	x[NMODEL*iblock + m] = xprior[m];
      }
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
