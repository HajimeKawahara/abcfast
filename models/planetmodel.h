/* #include <math.h> */
#define PCRIT 0.01
/* unit conversion */
#define Y2D 365.242189
#define RSOLAU 0.00464912633
#define REARTHRSOL 0.009164

#define PI 3.141592653589793
#define SIGMAHK 0.03
#define MESthreshold 7.1
/* limb darkening - Delta_max relation for G-type star (u=0.6) from Burke+2015 */
#define Cld 1.0874
#define Sld 1.0187

#define Adet 4.65
#define Bdet 0.98 

#define FAC0 0.014970127150014402
#define FAC1 0.002649580026551221
#define FAC2 0.00039843308669943166
#define FAC3 5.208275643129827e-05
#define FAC4 6.021127911132747e-06
#define FAC5 6.239510788738596e-07
#define FAC6 5.8586955762803736e-08
#define FAC7 5.02892324144238e-09
#define FAC8 3.975433392444569e-10
#define FAC9 2.9124054157103066e-11
#define FAC10 1.9879900448534514e-12
#define FAC11 1.2702811788201033e-13
#define FAC12 7.629316389309933e-15
#define FAC13 4.322558860798827e-16
#define FAC14 2.3177259307232317e-17
#define FAC15 1.1795042904443933e-18
#define FAC16 5.711885183750088e-20
#define FAC17 2.6382841495381463e-21
#define FAC18 1.164805364034502e-22
#define FAC19 4.925181243274852e-24
#define MAXX 12.822822822822824

#include "genpoisson.h"

/* 
pow(b,a)=0.98**4.65=0.9103350239605226
>>> scipy.special.gamma(4.65)
14.365526844616904
*/

__device__ void model(float* Ysim, float* param, curandState* s, float* aux, int isample){
  
  /* aux = stellar radius, CDPP, observing duration */
  float rstar;
  float mstar;
  float sigCDPP;
  float MESthre;
  float Tdur;
  float fduty;
  
  float abscosi;
  float arsol;
  
  int Npick;
  int ipick;
  int istar;
  
  float rl0;
  float rl1;
  
  /* random gen */
  float e;
  float omega;
  float P;
  float Rp;
  float k;
  
  float M;
  float MES;
  float X;
  float Delta;
  float pdet;
  float bX;
  float Ntrn;
  float onemf;
  float pwin;
  
  float indicator;
  float ppick;
  
  /* presampling from Nstarss with p(cosi < ccrit) = ccrit  */
  /* see ipynb Bonomial... */
  if(threadIdx.x == 0){
    ppick = PCRIT*param[0];

    if( ppick > PCHANGE ){
      /* using Gaussian approx */
      Npick=curand_normal(s)*ppick*(1.0-ppick)*Nstars + Nstars*ppick; 
    }else{
      /* using Poisson approx */
      Npick=poissonf(ppick*float(Nstars),s);
    }
    cache[NRESERVED] = Npick;
    /* printf("lambda=%2.8f Npick=%d \n",ppick*float(Nstars), Npick); */
    
  }
  __syncthreads();

  Npick = cache[NRESERVED];


  
  /* initialization for each thread */
  Ysim[0]=0.0;
    
  for (int p=0; p<int(float(Npick-1)/float(blockDim.x))+1; p++){
    
    if(p*blockDim.x+isample < Npick){
      
      /* stellar/lc parameters */
      istar = int(curand_uniform(s)*float(Nstars));
      rstar = aux[istar];
      mstar = aux[istar + Nstars];
      sigCDPP = aux[istar + 2*Nstars];
      MESthre = aux[istar + 3*Nstars];
      Tdur = aux[istar + 4*Nstars];
      fduty = aux[istar + 5*Nstars];

      /*      if(threadIdx.x == 0){
	printf("istar=%d rstar=%2.8f mstar=%2.8f sigCDPP=%2.8f MESthre=%2.8f Tdur=%2.8f fduty=%2.8f \n",istar,rstar,mstar,sigCDPP,MESthre,Tdur,fduty);
	} */
      
      /* |cosi| sim U(0,1) but restricted to < pcrit */
      abscosi = curand_uniform(s)*PCRIT;
      omega = curand_uniform(s)*2.0*PI;
      P = exp(logPmin + curand_uniform(s)*(logPmax - logPmin));
      Rp = exp(logRpmin + curand_uniform(s)*(logRpmax - logRpmin));
      arsol = pow(P/Y2D,2.0/3.0)/pow(mstar,1.0/3.0)/RSOLAU;

      /*      if(threadIdx.x == 0){
	printf("abscosi=%2.8f omega=%2.8f P=%2.8f Rp=%2.8f arsol=%2.8f \n",abscosi,omega,P,Rp,arsol);
	} */
      
      M = Tdur/P;
      Ntrn = M*fduty;
      k = Rp*REARTHRSOL/rstar;

      /*      if(threadIdx.x == 0){
	printf("M=%2.8f Tdur=%2.8f P=%2.8f Ntrn=%2.8f k=%2.8f \n",M,Tdur,P,Ntrn,k);
	} 
      */
      
      /* eccentricity according to the Rayleigh distribution */
      rl0 = curand_normal(s);
      rl1 = curand_normal(s);    
      e = min(0.999, SIGMAHK*sqrt(pow(rl0,2)+pow(rl1,2)) );

      /* if(threadIdx.x == 0){
	printf("e=%2.8f \n",e);
	} */
      
      Delta = 0.84*(Cld + Sld*k)*pow(k,2);
      
      /* p_det (Gamma distribution) */
      MES = sqrt(Ntrn)*Delta*1.e6/sigCDPP;
      X = max(0.0,MES - 4.1 - (MESthre - 7.1));

      /*      if(threadIdx.x == 0){
	printf("Delta=%2.8f sigCDPP=%2.8f MES=%2.8f X=%2.8f \n",Delta, sigCDPP, MES, X,k );
	} */
      


      
      /* CDF of the Gamma distribution is P(a, bX) to derive it, see (5.5) in Temme 1994 see comp_gammafac.py for precision */
      bX = Bdet*X;
      pdet = FAC0*exp(-bX)*pow(bX,0) + FAC1*exp(-bX)*pow(bX,1) + 
	FAC2*exp(-bX)*pow(bX,2) + FAC3*exp(-bX)*pow(bX,3) + 
	FAC4*exp(-bX)*pow(bX,4) + FAC5*exp(-bX)*pow(bX,5) + 
	FAC6*exp(-bX)*pow(bX,6) + FAC7*exp(-bX)*pow(bX,7) + 
	FAC8*exp(-bX)*pow(bX,8) + FAC9*exp(-bX)*pow(bX,9) +
	FAC10*exp(-bX)*pow(bX,10) + FAC11*exp(-bX)*pow(bX,11) + 
	FAC12*exp(-bX)*pow(bX,12) + FAC13*exp(-bX)*pow(bX,13) + 
	FAC14*exp(-bX)*pow(bX,14) + FAC15*exp(-bX)*pow(bX,15) + 
	FAC16*exp(-bX)*pow(bX,16) + FAC17*exp(-bX)*pow(bX,17) + 
	FAC18*exp(-bX)*pow(bX,18) + FAC19*exp(-bX)*pow(bX,19);
      pdet = pdet*pow(bX,Adet);
      
      if(bX > MAXX){
	pdet = 1.0;
      }

      /* p_win */
      onemf = 1.0 - fduty;
      pwin = 1.0 - pow(onemf,M) - M*fduty*pow(onemf,M - 1.0) - 0.5*M*(M-1.0)*pow(fduty,2)*pow(onemf,M-2.0);
      pwin = max(pwin,0.0);

      /*      if(threadIdx.x == 0){
	printf("pdet=%2.8f pwin=%2.8f M=%2.8f fduty=%2.8f \n",pdet,pwin,M,fduty);
	} */

      
      indicator=1.0;
      /* Check detection probability */
      if( pwin*pdet  < curand_uniform(s)){
	indicator = 0.0;
      }
      
      /* Check transit or not */
      if ( rstar*(1.0+e*sin(omega)) < arsol*abscosi*(1.0-pow(e,2))){
	indicator = 0.0;
      }
      
      Ysim[0]=Ysim[0]+indicator;
      
    }
  }
}

