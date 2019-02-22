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
    
    for (int m=0; m<npart/nthread; m++){

    ipart = m*nthread+ithread;
    /* computing qf (Gaussian transition kernel) */
    qf=0.0;
    for (int m=0; m<NMODEL; m++){
      qf =+ -0.5*pow((xprev[NMODEL*ipart + m] - xnew[NMODEL*iblock + m]),2)/invcov[m];
	}
    qf = exp(qf);
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
