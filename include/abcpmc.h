/* abcpmc */
/* i=0,...,n-1, are X[i], i=n,...,n+npart-1, are previous X[i] i=n+npart is used for a block prior (xast) */
#include "genalias.h"

extern "C"{

  __global__ void abcpmc(float* x, float* xprev, float* Ysm, float epsilon, int* Ki, int* Li, float* Ui, float* Qmat, int seed, float* dist, int* ntry, int ptwo){

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
    float xmodel[NDATA];
    float rn[NMODEL];
    
    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRYX){
      if(ithread==0){
	printf("EXCEED MAXTRYX. iblock=%d \\n",iblock);
	for (int m=0; m<NMODEL; m++){
	    x[NMODEL*iblock + m] = CUDART_NAN_F;
	}
	ntry[iblock]=MAXTRYX;	  
      }
      return;
    }
    
    /* sampling a prior from the previous posterior*/
    if(ithread == 0){
      isel=aliasgen(Ki, Li, Ui, npart,&s);
      for (int m=0; m<NMODEL; m++){
	rn[m] = curand_normal(&s);
      }

      for (int m=0; m<NMODEL; m++){
	xprior[m]=xprev[NMODEL*isel+m];
	
	for (int k=0; k<NMODEL; k++){
	  xprior[m] += Qmat[m*NMODEL+k]*rn[k];
	}
	
	cache[NDATA*n+m] = xprior[m];
      }
    }
    __syncthreads();
    /* ===================================================== */
    for (int m=0; m<NMODEL; m++){
      xprior[m] = cache[NDATA*n+m];
    }

    model(xprior, xmodel, &s);
    for (int m=0; m<NDATA; m++){
      cache[NDATA*ithread+m] = xmodel[m];
    }

    __syncthreads();
    /* ===================================================== */
    
    /* ----------------------------------------------------- */
    /* SUMMARY STATISTICS */
    /* sum of |X - Y|/n */
    /* thread cooperating computation of rho */
    int i = ptwo;
    while(i !=0) {
      if ( ithread + i < n && ithread < i){
	for (int m=0; m<NDATA; m++){
	  cache[NDATA*ithread + m] += cache[NDATA*(ithread + i) + m];
	}
      }
        __syncthreads();
        i /= 2;
    }
    
    __syncthreads();
    /* ===================================================== */
    rho = 0.0;
    for (int m=0; m<NDATA; m++){
      rho += abs(cache[m] - Ysm[m])/n;     
    }

    /* ----------------------------------------------------- */
    
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
