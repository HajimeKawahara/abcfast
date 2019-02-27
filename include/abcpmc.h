/* abcpmc */
/* i=0,...,n-1, are X[i], i=n,...,n+npart-1, are previous X[i] i=n+npart is used for a block prior (xast) */
#include "genalias.h"

extern "C"{

  __global__ void abcpmc(float* x, float* xprev, float* Ysm, float epsilon, int* Ki, int* Li, float* Ui, float* Qmat, int seed, float* dist, int* ntry, int ptwo){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int nthread = blockDim.x;
    int npart = gridDim.x;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*NSAMPLE + ithread;
    float uni;
    int isel;
    int isample;
    float xprior[NMODEL];
    float xmodel[NDATA];
    float rn[NMODEL];
    
    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRYX){
      if(ithread==0){
	printf("EXCEED MAXTRYX. iblock=%d \n",iblock);
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
	
	cache[NDATA*NSAMPLE+m] = xprior[m];
      }
    }
    __syncthreads();
    /* ===================================================== */
    for (int m=0; m<NMODEL; m++){
      xprior[m] = cache[NDATA*NSAMPLE+m];
    }

    for (int p=0; p<int(float(NSAMPLE-1)/float(nthread))+1; p++){
      isample = p*nthread + ithread;
      model(xprior, xmodel, &s);
      for (int m=0; m<NDATA; m++){
	cache[NDATA*isample+m] = xmodel[m];
      }
    }

    __syncthreads();
    /* ===================================================== */
    
    /* ----------------------------------------------------- */
    /* SUMMARY STATISTICS */
    /* sum of |X - Y|/n */
    /* thread cooperating computation of rho */
    int i = ptwo;
    while(i !=0) {

      for (int p=0; p<int(float(NSAMPLE-1)/float(nthread))+1; p++){
	isample = p*nthread + ithread;
	
	if ( isample + i < NSAMPLE && ithread < i){
	  for (int m=0; m<NDATA; m++){
	    cache[NDATA*isample + m] += cache[NDATA*(isample + i) + m];
	  }
	}
      }
      
        __syncthreads();
        i /= 2;
    }
    
    __syncthreads();
    /* ===================================================== */
    rho = 0.0;
    for (int m=0; m<NDATA; m++){
      rho += abs(cache[m] - Ysm[m])/NSAMPLE;     
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
