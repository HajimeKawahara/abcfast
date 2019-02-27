/* stucture of shared memory */
/* abcpmc_init */
/* k=0,...,(NSAMPLE-1)*NDATA, are X[i,j]=X[k], k=2*i + j, k=NSAMPLE*NDATA,...,NSAMPLE*nd+(NMODEL-1) is used for a block prior (xast) */

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
	  cache[NSAMPLE+m] = parmodel[m];
	}
	
      }
      
      __syncthreads();
      /* ===================================================== */
      for (int p=0; p<int(float(NSAMPLE-1)/float(nthread))+1; p++){
	isample = p*nthread + ithread;
	if (isample < NSAMPLE){
	  
	  for (int m=0; m<NMODEL; m++){
	    parmodel[m] = cache[NSAMPLE+m];
	  }
	  
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
