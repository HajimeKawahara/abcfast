#define NMAXPOI 10000
#include <curand_kernel.h>

__device__ int poissonf(float p, curandState* s){

  float c;
  int m;
  float d;
  float pu;
  float pl;
  int Xu;
  int Xl;
  float V;
  float U;
  int i;
  
  c = 1.0/p;
  m = int(p);
  d = - p + m*log(p);
  
  for (int k=0; k<m; k++){
    d = d - log(float(k+1));
  }
  
  d = exp(d);
  pu=d;
  pl=d;
  Xu=m;
  Xl=m;
  
  i=0;
  V = curand_uniform(s) - pu;
  
  while(i < NMAXPOI){
    i=i+1;
    
    if(V <= 0.0){
      return Xu;
    }
    
    U = V;
    if(Xl > 0.0){
      pl=pl*c*float(Xl);
      Xl = Xl - 1;
      V = U - pl;
      
      if(V < 0.0){
	return Xl;
      }
      U = V;
    }      
    Xu = Xu+1;
    pu = pu*p/float(Xu);
    V = U - pu;      
  }
  
  return -1;
  
}
