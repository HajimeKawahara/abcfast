#include <curand_kernel.h>

__device__ void gennorm2df(float* x,float* hparam, curandState* s){
  /* 2D gaussian with covariance matrix = [[a,c],[c,b]] */
  /* and mean = [mu0, mu1] */
  float mu0 = hparam[0];
  float mu1 = hparam[1];
  float a = hparam[2];
  float b = hparam[3];
  float c = hparam[4];

  float norm;
  float e11;
  float e12;
  float e21;
  float e22;
  float rn1 = curand_normal(s);
  float rn2 = curand_normal(s);

  if(c==0.0){
    x[0] = rn1*sqrt(a) + mu0;
    x[1] = rn2*sqrt(b) + mu1;

  }else{
    float a2 = pow(a,2);
    float b2 = pow(b,2);
    float c2 = pow(c,2);
    
    float fac = sqrt(a2 - 2*a*b + b2 + 4*c2)/2;  
    float sqrt_lam1 = sqrt(a/2 + b/2 - fac);        
    float sqrt_lam2 = sqrt(a/2 + b/2 + fac);
    
    e11 = -c/(a/2 - b/2 + fac);
    norm = sqrt(pow(e11,2)+1.0);
    e11 = sqrt_lam1*e11/norm;
    e12 = sqrt_lam1/norm;
    
    e21 = -c/(a/2 - b/2 - fac);
    norm = sqrt(pow(e21,2)+1.0);
    e21 = sqrt_lam2*e21/norm;
    e22 = sqrt_lam2/norm;
    
    x[0] = e11*rn1 + e21*rn2 + mu0;
    x[1] = e12*rn1 + e22*rn2 + mu1;
  }
  /* printf("S %2.8f %2.8f \n",x[0],x[1]); */
}
