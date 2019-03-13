/* Hierarchical ABC */
/* habcpmc */

/* 
In this model, parhyper is fixed.
- the prior marameters (hparam) is common in a grid.
- NSUBJECT sets of the model parameters (param) in a grid.

hparam(NHPARAM) ~ hyperprior( parhyper(NHYPER) )
param(NPARAM) ~ prior( hparam(NHPARAM) )
Ysim(NDATA) ~ model( param(NPARAM) )

NHYPER: dimension of the hyperprior parameter (parhyper)
NHPARAM: dimension of the prior parameters (hparam)
NPARAM: dimension of the model parameters (param)
NSUBJECT: number of the subjects
NDATA: number of the output parameters of a model
NSAMPLE: number of the samples in a dataset

- cache[k]
k= +NDATA*NSAMPLE for sumulated data Ysim, +NHPARAM for the prior parmeters, +NSUBJECT*NPARAM for the model parameters

- isample :-> isubject = mod(isample,NSS) for thread cooperation. NSS=N sample/subject

*/

#include "genalias.h"

extern "C"{
  __global__ void habcpmc(float* x, float* xprev, float* Ysm, float epsilon, int* Ki, int* Li, float* Ui, float* Qmat, int seed, float* dist, int* ntry){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int nthread = blockDim.x;
    int npart = gridDim.x;

    float uni;
    int isel;
    int isample;
    int isubject;
    int isubdat;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*nthread + ithread;
    float param[NPARAM];
    float Ysim[NDATA];
    float hparam[NHPARAM];
    float rn[NHPARAM];
    
    curand_init(seed, id, 0, &s);
    
    for ( ; ; ){


    /* limitter */
      cnt++;
      if(cnt > MAXTRYX){
	
	if(ithread==0){
	  printf("EXCEED MAXTRYX. iblock=%d \n",iblock);

	  for (int m=0; m<NHPARAM; m++){
	    x[NHPARAM*iblock + m] = CUDART_NAN_F;
	  }
	ntry[iblock]=MAXTRYX;
        
	}
	return;
      }

      /* sampling a hyperprior from the previous (hyper)posterior*/
      if(ithread == 0){
	isel=aliasgen(Ki, Li, Ui, npart,&s);
	for (int m=0; m<NHPARAM; m++){
	  rn[m] = curand_normal(&s);
	}
	
	for (int m=0; m<NHPARAM; m++){
	  hparam[m]=xprev[NHPARAM*isel+m];

	  for (int k=0; k<NHPARAM; k++){
	    hparam[m] += Qmat[m*NHPARAM+k]*rn[k];
	    
	  }
	  
	  cache[NDATA*NSAMPLE+m] = hparam[m];
	}
      }
      __syncthreads();

      /* ===================================================== */
      /* sampling from a hyper prior, getting the hyperparameter (x NHPARAM elements) */
      if(ithread == 0){
	hyperprior(hparam, &s);
	for (int m=0; m<NHPARAM; m++){
	  cache[NDATA*NSAMPLE+m] = hparam[m];
	}
      }
      __syncthreads();
      /* ===================================================== */

      /* loading hyperparameters for each thread s*/
      for (int m=0; m<NHPARAM; m++){
	hparam[m] = cache[NDATA*NSAMPLE+m];
      }

      
      /* sampling from a prior, getting NSUBJECT sets of the model parameters (x NHPARAM elements) */
      for (int k=0; k<int(float(NSUBJECT-1)/float(nthread))+1; k++){
	isubject = k*nthread + ithread;

	if(isubject < NSUBJECT){
	  
	  prior(param,hparam, &s);

	  for (int m=0; m<NPARAM; m++){
	    cache[NDATA*NSAMPLE+NHPARAM+isubject*NPARAM + m] = param[m];
	  }
	  
	}	  
      }	
      
      __syncthreads();

      
      /* ===================================================== */
      for (int k=0; k<int(float(NSAMPLE-1)/float(nthread))+1; k++){
	isample = k*nthread + ithread;
	isubject = isample%NSUBJECT;
	/*	printf("NSAMPLE=%d, isample=%d, NSS=%d , isubjet=%d \n",NSAMPLE,isample,NSS,isubject); */
	
	if (isample < NSAMPLE && isubject < NSUBJECT){

	  /* param is not common, so must read for each isample */
	  for (int m=0; m<NPARAM; m++){
	    param[m] = cache[NDATA*NSAMPLE+NHPARAM+isubject*NPARAM + m];
	  }
	  
	  model(Ysim,param, &s);

	  for (int m=0; m<NDATA; m++){
	    cache[NDATA*isample+m] = Ysim[m];	    
	  }
	  
	}
      }
      
      __syncthreads();
      /*      printf("%2.8f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f NSAMPLE=%d\n",cache[0],cache[1],cache[2],cache[3],cache[4],cache[5],cache[6],cache[7],NSAMPLE); */

      /* ===================================================== */

      /* ===================================================== */
	  
      /* SUMMARY STATISTICS */
      /* sum of sum|X - Y|/ndata/nsubject */
      
      /* ----------------------------------------------------- */
      /* sum toward each observation (compute Ysm^sim = sum(Ysim) ) */

      int i = PNSS*NSUBJECT;

      while(i >= NSUBJECT) {
	
	for (int p=0; p<int(float(PNSS*NSUBJECT-1)/float(nthread))+1; p++){
	  isample = p*nthread + ithread;
	  /* isubject = isample%NSS; */
	  if (isample + i < NSAMPLE && isample < i){

	    /*	    if(cache[NDATA*(isample + i)]>100){
	      	      printf("NSUBJECT=%d,PNSS=%d,PNSS*NSUBJECT=%d NSAMPLE=%d ind=%d\n",NSUBJECT,PNSS,PNSS*NSUBJECT,NSAMPLE,NDATA*(isample + i));
	      printf("h0=%2.8f h1=%2.8f p0=%2.8f Ysim0x=%2.8f,f iblock=%d ithread=%d isample=%d i=%d ind=%d \n",hparam[0],hparam[1],param[0],cache[NDATA*(isample + i)],iblock,ithread,isample,i,NDATA*(isample + i)); 
	      } */

	    for (int m=0; m<NDATA; m++){
	      cache[NDATA*isample + m] += cache[NDATA*(isample + i) + m];
	    }
	    
	  }
	}
	
	__syncthreads();
	i /= 2;
      }
      
      __syncthreads();
      /*      printf("h0=%2.8f h1=%2.8f p0=%2.8f Ysmsub0=%2.8f,Ysmsub1=%2.8f,Ysmsub2=%2.8f,Ysmsub3=%2.8f iblock=%d ithread=%d \n",hparam[0],hparam[1],param[0],cache[0],cache[1],cache[2],cache[3],iblock,ithread); */

      /* ===================================================== */
      /* compute |Ysm^sim - Ysm| for each subject/idata */
      for (int k=0; k<int(float(NSUBJECT-1)/float(nthread))+1; k++){
	isubject = k*nthread + ithread;
	if(isubject < NSUBJECT){
	  for (int m=0; m<NDATA; m++){
	    cache[NDATA*isubject + m] = abs(cache[NDATA*isubject + m] - Ysm[NDATA*isubject + m]);
	  }
	}
      }
      
      __syncthreads();
      

      /* ===================================================== */
      /* thread coorporating sum toward subject and idata */

      i = PNSUBDAT;
      while(i != 0) {
	for (int k=0; k<int(float(PNSUBDAT-1)/float(nthread))+1; k++){
	  isubdat = k*nthread + ithread;	
	  if(isubdat + i < NSUBDAT && isubdat < i){
	    cache[isubdat] += cache[isubdat+i];
	  }
	}
	__syncthreads();
	i /= 2;
      }

      
      __syncthreads();
      /* ----------------------------------------------------- */
      
      /* ===================================================== */
      /* loading rho and judge it */
      /* ===================================================== */
      rho = cache[0]/NSAMPLE/NDATA;


      
      if(rho<epsilon){
	
	if(ithread<NHPARAM){
	    x[NHPARAM*iblock + ithread] = hparam[ithread];
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
