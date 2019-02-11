#include <curand_kernel.h>

__device__ int binomialf(int n,float p,curandState *s){
  /* Binomial model generator */
  
  int val=0;
  
  for (int i = 0; i < n; i++){
    if (curand_uniform(s) <= p){
      val++;
    }
  }
  
  return val;
  
}

