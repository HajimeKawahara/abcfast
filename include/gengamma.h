#include <curand_kernel.h>
#define MAXTRY 10000

__device__ float gammaf(float ain, float bin, curandState* s){

  unsigned long cnt=0;
  float d, c, y, v, u, a;

  if(ain > 1.0){
    a = ain;
  }else if(ain > 0.0){
    a = ain + 1.0;
  }else{
    return 0.0;
  }
  
  d = a - 1. / 3.;
  c = 1. / sqrt(9. * d);
  for (;;) {
    do {
      y =curand_normal(s);
      v = 1. + c * y;
    } while (v <= 0.);
    
    cnt=cnt+1;
    if(cnt > MAXTRY){
      printf("*** EXCEED MAXTRY ***");
      return 0.0;
    }
    
    v = v * v * v;
    u = curand_uniform(s);
    if (u < 1. - 0.0331 * (y * y) * (y * y)) {
      
      if(a > ain){
	return (d * v)*pow(curand_uniform(s),1.0/ain)/bin;
      }else{
	return (d * v)/bin;
      }
      
    }
    if (log(u) < 0.5 * y * y + d * (1. - v + log(v))) {
      
      if(a > ain){
	return (d * v)*pow(curand_uniform(s),1.0/ain)/bin;
      }else{
	return (d * v)/bin;
      }
      
    }
    
  }

}
 

