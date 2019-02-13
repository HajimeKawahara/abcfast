#include <curand_kernel.h>

__device__ int aliasgen(int* Ki, int* Li, float* Ui, int nt, curandState* s){
  
  float xuni;
  
  xuni = curand_uniform(s);    
  float pb = xuni * float(nt);
  int index = __float2int_rd(pb);
  
  if(xuni == 1.0){
    index = 0;
  }

  if(Ui[index] < pb - index){
    return Ki[index];
  }else{
    return Li[index];
  }
  
}
