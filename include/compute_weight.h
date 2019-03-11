extern "C"{

      __global__ void compute_weight(float* wnew, float* wprev, float* xnew, float* xprev, float* invcov){
    int nthread = blockDim.x;
    int npart = gridDim.x;
    float rnthread = float(nthread);
    float rnpart = float(npart);
    int ipart;

    /* prev:=t-1, new:=t in TVZ12 */ 
    /* iblock := i in TVZ12 */
    int iblock = blockIdx.x;

    /* ithread := j in TVZ12 */
    int ithread = threadIdx.x;
    float qf;
    
    for (int l=0; l<npart/nthread; l++){

      ipart = l*nthread+ithread;
      /* computing qf (Gaussian transition kernel) */
      qf=0.0;
      for (int m=0; m<NWPARAM; m++){
	for (int k=0; k<NWPARAM; k++){
	  /* [NWPARAM*ipart + m] - xnew[NWPARAM*iblock + m] */
	  /* xprev: ipart,imodel => NWPARAM*ipart + imodel  */
	  /* xnew: iblock,imodel => NWPARAM*iblock + imodel  */
	  /* imodel = m,k  */
	  /* here, computing an element of a quadratic form x_m x_k A_mk */
	  /* A_mk := m*NWPARAM + k = k*NWPARAM + m because inverse covariance matrix is a symmetric matrix */

	  qf += (xprev[NWPARAM*ipart + m] - xnew[NWPARAM*iblock + m])*(xprev[NWPARAM*ipart + k] - xnew[NWPARAM*iblock + k])*invcov[m*NWPARAM + k];
	  
	}
      }
      /*      printf("xqf=%2.8f qf= %2.8f 00=%2.8f, 01=%2.8f, 10=%2.8f, 11=%2.8f, 00=%2.8f, 01=%2.8f, 10=%2.8f, 11=%2.8f \n ",qf,exp(-0.5*qf),(xprev[NWPARAM*ipart + 0] - xnew[NWPARAM*iblock + 0])*(xprev[NWPARAM*ipart + 0] - xnew[NWPARAM*iblock + 0]),(xprev[NWPARAM*ipart + 0] - xnew[NWPARAM*iblock + 0])*(xprev[NWPARAM*ipart + 1] - xnew[NWPARAM*iblock + 1]),(xprev[NWPARAM*ipart + 1] - xnew[NWPARAM*iblock + 1])*(xprev[NWPARAM*ipart + 0] - xnew[NWPARAM*iblock + 0]),(xprev[NWPARAM*ipart + 1] - xnew[NWPARAM*iblock + 1])*(xprev[NWPARAM*ipart + 1] - xnew[NWPARAM*iblock + 1]),invcov[0*NWPARAM + 0],invcov[0*NWPARAM + 1],invcov[1*NWPARAM + 0],invcov[1*NWPARAM + 1]); */

      qf = exp(-0.5*qf);
      /* thread cooperating computation of a denominater */        
      cache[ipart] = wprev[ipart]*qf;

    }
    __syncthreads();

    int i = npart/2;
    while(i !=0) {

    for (int m=0; m<npart/nthread; m++){

    ipart = m*nthread+ithread;
    if (ipart < i){
    cache[ipart] += cache[ipart + i];
    }
    __syncthreads();
    i /= 2;

    }
    }

    __syncthreads();
    /* ===================================================== */

    if (ithread==0){
    /* computing weight */
    wnew[iblock] = cache[0];

    return;

    }else{

    return;

    }


    }


    }
