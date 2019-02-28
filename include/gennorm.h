#include <curand_kernel.h>

__device__ float normf(float mu, float sigma, curandState* s){
  
  return  curand_normal(s)*sigma + mu;
    
}
