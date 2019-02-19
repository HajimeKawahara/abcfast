/* stucture of shared memory */
/* abcpmc_init */
/* i=0,...,n-1, are X[i], i=n is used for a block prior (xast) */

extern "C"{
  __global__ void abcpmc_init(float* x, float Ysum, float epsilon, int seed, float* parprior, float* dist, int* ntry, int ptwo){

    curandState s;
    int cnt = 0;
    float p;
    float rho;
    int n = blockDim.x;
    int iblock = blockIdx.x;
    int ithread = threadIdx.x;
    unsigned long id = iblock*n + ithread;
    float xprior[NMODEL];
    
    curand_init(seed, id, 0, &s);

    for ( ; ; ){

    /* limitter */
      cnt++;
      if(cnt > MAXTRYX){
	
	if(ithread==0){
	  printf("EXCEED MAXTRYX. iblock=%d \n",iblock);
	  for (int m=0; m<NMODEL; m++){
	    x[NMODEL*iblock + m] = -1.0;
	  }
	ntry[iblock]=MAXTRYX;
        
	}
	return;
      }
      
      /* sampling a prior from the Gamma distribution */
      if(ithread == 0){
	prior(parprior, xprior, &s);
	for (int m=0; m<NMODEL; m++){
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
      
      
      
      /* thread cooperating computation of rho */
      int i = ptwo;
      /*int i = n/2;*/
      while(i !=0) {
        if ( ithread + i < n && ithread < i){
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
