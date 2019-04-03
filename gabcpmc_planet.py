from gabc.gabcpmc import *
from gabc.utils.statutils import *
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from scipy.stats import gamma as gammafunc
    from scipy.stats import norm as normfunc
    import time
    import sys
    import pandas as pd

    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC PMC Method.")
    print("This code demonstrates the Kepler planet occurence inference")
    print("*******************************************")

    #preparing data
    nsample=500
    lambda_true=0.1
    Yobs=random.exponential(1.0/lambda_true,nsample)

    # start ABCpmc 
    abc=ABCpmc()
    abc.maxtryx=1000
    abc.npart=512
    abc.wide=2.0

    
    # input model/prior

    ## input aux from the stellar catalog
    planet_data=pd.read_csv("/home/kawahara/exocal/exosnow/data/kepler.csv")
    rstar=planet_data["radiusnew"].values
    mstar=planet_data["mass"].values
    sigCDPP=planet_data["rrmscdpp04p5"].values
    mesthre=planet_data["mesthres04p5"].values
    Tdur = planet_data["dataspan"].values
    fduty = planet_data["dutycycle"].values
    
    mask=(sigCDPP==sigCDPP)&(mstar>0.0)&(rstar>0.0)&(mesthre==mesthre)&(Tdur==Tdur)&(fduty==fduty)

    #SELECT Main-Sequence
    teff=planet_data["teff"].values
    logg=planet_data["logg"].values

    mask=mask&(teff<7000.0)&(teff>4000.0)&(logg>4.0)
    
    rstar=rstar[mask]
    mstar=mstar[mask]
    sigCDPP=sigCDPP[mask]
    mesthre=mesthre[mask]
    Tdur=Tdur[mask]
    fduty=fduty[mask]
    
    nstar=len(rstar)
    
    abc.aux=np.concatenate([rstar,mstar,sigCDPP,mesthre,Tdur,fduty])
    print("NSTAR=",nstar)
    print("Do not forget to include errors of Rstar in future.")

    Pmin=10.0
    Pmax=20.0
    Rpmin=1.0
    Rpmax=1.25
    
    logPmin=np.log(Pmin)
    logPmax=np.log(Pmax)
    logRpmin=np.log(Rpmin)
    logRpmax=np.log(Rpmax)

    pchange=0.01458*(nstar/100000)**-0.3333
    
    abc.nparam=1
    abc.ntcommon=1 #use 1 thread common value in shared memory for Npick
    abc.model=\
    "#define Nstars "+str(nstar)+"\n"\
    +"#define logPmin "+str(logPmin)+"\n"\
    +"#define logPmax "+str(logPmax)+"\n"\
    +"#define logRpmin "+str(logRpmin)+"\n"\
    +"#define logRpmax "+str(logRpmax)+"\n"\
    +"#define PCHANGE "+str(pchange)+"\n"\
    +""" 

    #include "planetmodel.h"

    """
    
    # prior 
    fmax=1.0
    
    def fprior():
        def f(x):            
            return 1.0/fmax
        return f
    abc.fprior = fprior()#

    abc.prior=\
    "#define FMAX "+str(fmax)+"\n"\
    """
    #include "gengamma.h"

    __device__ void prior(float* param,curandState* s){

    param[0] = curand_uniform(s)*FMAX;

    return;

    }
    """

    # data and the summary statistics
    abc.nsample = 512
    abc.ndata = 1
    observed_data=pd.read_csv("/home/kawahara/exocal/exosnow/data/q1_q17_dr25_koi.csv",delimiter=",",comment="#")
    Pkoi=observed_data["koi_period"]
    Rpkoi=observed_data["koi_prad"]
    mask=observed_data["koi_pdisposition"]=="CANDIDATE"
    mask=mask&(Pkoi>Pmin)&(Pkoi<Pmax)&(Rpkoi>Rpmin)&(Rpkoi<Rpmax)
    
    Yobs=len(observed_data[mask])

    Ysum = np.float32(Yobs)
    abc.Ysm = np.array([Ysum])
    
    #set prior parameters
    abc.epsilon_list = np.array([0.007,0.005,0.003,0.001,0.0005])

    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()
    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()

    tend = time.time()

    print(tend-tstart,"sec")

    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(111)
    plt.hist(abc.x,bins=30,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
    plt.show()

