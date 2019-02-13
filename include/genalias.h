#include <curand_kernel.h>

__device__ int aliasgen(int* Ki, int* Li, float* Ui, int nt, curandState* s){
  
  float xuni;
  
  xuni = 1.0 - curand_uniform(s);
  
  float pb = xuni * float(nt);
  int index = __float2int_rd(pb);
  
  if(Ui[index] < pb - index){
    return Ki[index];
  }else{
    return Li[index];
  }
  
}
