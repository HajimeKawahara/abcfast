/* abcpmc */
/* 
In this model, parhyper is fixed.
- the prior marameters (hparam) is fixed.
- the model parameters (param) is common in a grid.

param(NPARAM) ~ prior( hparam(NHPARAM) )
Ysim(NDATA) ~ model( param(NPARAM) )

NHPARAM: dimension of the prior parameters (hparam)
NPARAM: dimension of the model parameters (param)
NDATA: number of the output parameters of a model
NSAMPLE: number of the samples in a dataset

- cache[k]
k= +NDATA*NSAMPLE for sumulated data Ysim, +NPARAM for the model parameters

*/

#include "genalias.h"

extern "C"{

  __global__ void abcpmc(float* x, float* xprev, float* Ysm, float epsilon, int* Ki, int* Li, float* Ui, float* Qmat, int seed, float* dist, int* ntry, float* aux){

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
    float param[NPARAM];
    float Ysim[NDATA];
    float rn[NPARAM];

    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
    cnt++;
    if(cnt > MAXTRYX){
      if(ithread==0){
	printf("EXCEED MAXTRYX. iblock=%d \n",iblock);
	for (int m=0; m<NPARAM; m++){
	    x[NPARAM*iblock + m] = CUDART_NAN_F;
	}
	ntry[iblock]=MAXTRYX;	  
      }
      return;
    }
    
    /* sampling a prior from the previous posterior*/
    if(ithread == 0){
      isel=aliasgen(Ki, Li, Ui, npart,&s);
      for (int m=0; m<NPARAM; m++){
	rn[m] = curand_normal(&s);
      }

      for (int m=0; m<NPARAM; m++){
	param[m]=xprev[NPARAM*isel+m];
	
	for (int k=0; k<NPARAM; k++){
	  param[m] += Qmat[m*NPARAM+k]*rn[k];
	}
	
	cache[NDATA*NSAMPLE+m] = param[m];
      }
    }
    __syncthreads();
    /* ===================================================== */
    
    for (int m=0; m<NPARAM; m++){
      param[m] = cache[NDATA*NSAMPLE+m];
    } 
    
    for (int p=0; p<int(float(NSAMPLE-1)/float(nthread))+1; p++){
      isample = p*nthread + ithread;
      if(isample < NSAMPLE){
	
	model(Ysim, param, &s, aux, isample);
	for (int m=0; m<NDATA; m++){
	  cache[NDATA*isample+m] = Ysim[m];
	}
	
      }
    }

    __syncthreads();
    /* ===================================================== */
    
    /* ----------------------------------------------------- */
    /* SUMMARY STATISTICS */
    /* sum of |X - Y|/n */
    /* thread cooperating computation of rho */
    int i = PNSAMPLE;
    while(i !=0) {

      for (int p=0; p<int(float(PNSAMPLE-1)/float(nthread))+1; p++){
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
      
      if(ithread<NPARAM){
	x[NPARAM*iblock + ithread] = param[ithread];
      }	
      if(ithread==0){
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
