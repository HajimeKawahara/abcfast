/* stucture of shared memory */
/* abcpmc_init */
/* k=0,...,(n-1)*nd, are X[i,j]=X[k], k=2*i + j, k=n*nd,...,n*nd+(NMODEL-1) is used for a block prior (xast) */

extern "C"{
  __global__ void abcpmc_init(float* x, float* Ysm, float epsilon, int seed, float* parprior, float* dist, int* ntry, int ptwo){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int n = blockDim.x;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*n + ithread;
    float xprior[NMODEL];
    float xmodel[NDATA];
    
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
	prior(parprior, xprior, &s);
	
	/*	printf("ithread=%d val=%2.8f %2.8f %2.8f %2.8f \n", ithread, xmodel[0],xmodel[1], xprior[0],xprior[1]); */

	for (int m=0; m<NMODEL; m++){
	  cache[n+m] = xprior[m];
	}
	
      }

      
      __syncthreads();
      /* ===================================================== */
      
      for (int m=0; m<NMODEL; m++){
	xprior[m] = cache[n+m];
      }
      
      model(xprior, xmodel, &s);

      for (int m=0; m<NDATA; m++){
	cache[NDATA*ithread+m] = xmodel[m];

      }
	
      __syncthreads();
      /* ===================================================== */

      /* ----------------------------------------------------- */
      /* SUMMARY STATISTICS */
      /* sum of |X - Y|/ns */
      /* thread cooperating computation of rho */



      int i = ptwo;
      while(i !=0) {
        if ( ithread + i < n && ithread < i){
	  for (int m=0; m<NDATA; m++){
	    /*	    printf("ithread=%d m=%d F=%d L=%d valf=%2.8f vall=%2.8f \n", ithread, m, NDATA*ithread + m, NDATA*(ithread + i) + m, cache[NDATA*ithread + m],cache[NDATA*(ithread + i) + m]); */

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
	rho =+ abs(cache[m] - Ysm[m])/n;
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
