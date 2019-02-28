/* stucture of shared memory */
/* abcpmc_init */

/* 
In this model, parhyper is fixed.
- the prior marameters (parprior) is fixed.
- the model parameters (parmodel) is common in a grid.

parmodel(NMODEL) ~ prior( parprior(NPRIOR) )
Ysim(NDATA) ~ model( parmodel(NMODEL) )

NPRIOR: dimension of the prior parameters (parprior)
NMODEL: dimension of the model parameters (parmodel)
NDATA: number of the output parameters of a model
NSAMPLE: number of the samples in a dataset

- cache[k]
k= +NDATA*NSAMPLE for sumulated data Ysim, +NMODEL for the model parameters

*/


extern "C"{
  __global__ void abcpmc_init(float* x, float* Ysm, float epsilon, int seed, float* parprior, float* dist, int* ntry, int ptwo){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int nthread = blockDim.x;
    int isample;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*nthread + ithread;
    float parmodel[NMODEL];
    float Ysim[NDATA];
    
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
      
      /* sampling a prior */
      if(ithread == 0){
	prior(parprior, parmodel, &s);
	
	for (int m=0; m<NMODEL; m++){
	  cache[NDATA*NSAMPLE+m] = parmodel[m];
	}
	
      }
      
      __syncthreads();
      /* ===================================================== */
      for (int m=0; m<NMODEL; m++){
	parmodel[m] = cache[NDATA*NSAMPLE+m];
      }

      for (int p=0; p<int(float(NSAMPLE-1)/float(nthread))+1; p++){
	isample = p*nthread + ithread;
	if (isample < NSAMPLE){	  	  

	  model(parmodel, Ysim, &s);	  
	  for (int m=0; m<NDATA; m++){
	    cache[NDATA*isample+m] = Ysim[m];	    
	  }
	  
	}
      }
      
      __syncthreads();
      /* ===================================================== */
	        
      /* ----------------------------------------------------- */
      /* SUMMARY STATISTICS */
      /* sum of |X - Y|/ns */
      /* thread cooperating computation of rho */

      int i = ptwo;
      while(i !=0) {
	
	for (int p=0; p<int(float(NSAMPLE-1)/float(nthread))+1; p++){
	  isample = p*nthread + ithread;
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
      rho = 0.0;
      for (int m=0; m<NDATA; m++){
	rho += abs(cache[m] - Ysm[m])/NSAMPLE;
      }      
      /* ----------------------------------------------------- */
      
      if(rho<epsilon){
	
	if(ithread==0){

	  for (int m=0; m<NMODEL; m++){
	    x[NMODEL*iblock + m] = parmodel[m];
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
