/* Hierarchical ABC */
/* habcpmc_init */

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

extern "C"{
  __global__ void habcpmc_init(float* x, float* Ysm, float epsilon, int seed, float* dist, int* ntry){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int nthread = blockDim.x;
    int isample;
    int isubject;
    int isubdat;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*nthread + ithread;
    float param[NPARAM];
    float Ysim[NDATA];
    float hparam[NHPARAM];
    
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
	isubject = isample%NSS;
	
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
      /* ===================================================== */
	  
      /* SUMMARY STATISTICS */
      /* sum of sum|X - Y|/ndata/nsubject */
      
      /* ----------------------------------------------------- */
      /* sum toward each observation (compute Ysm^sim = sum(Ysim) ) */

      int i = PNSS*NSUBJECT;

      while(i >= NSUBJECT) {
	
	for (int p=0; p<int(float(PNSS*NSUBJECT-1)/float(nthread))+1; p++){
	  isample = p*nthread + ithread;
	  isubject = isample%NSS;
	  if (isample + i < NSAMPLE && isample < i){	   
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
      rho = cache[0]/NSUBJECT/NDATA/NSS;
      
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
