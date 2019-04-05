from abcfast.gabcpmc import *
from abcfast.utils import statutils

def ABCfrp(Pmin=256.0,Pmax=500.0,Rpmin=1.75,Rpmax=2.0):
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    import time
    import sys
    import pandas as pd

    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC PMC Method.")
    print("This code demonstrates the Kepler planet occurence inference")
    print("*******************************************")


    # start ABCpmc 
    abc=ABCpmc()
    abc.maxtryx=10000
    abc.npart=256
    abc.wide=2.0

    # data and the summary statistics
    abc.nsample = 512
    abc.ndata = 1
    observed_data=pd.read_csv("data/koi_berger.csv",delimiter=",")
    #observed_data=pd.read_csv("/home/kawahara/exocal/exosnow/data/q1_q17_dr25_koi.csv",delimiter=",",comment="#")
    
    Pkoi=observed_data["koi_period"]
    Rpkoi=observed_data["rpgaia"]
    #Rpkoi=observed_data["koi_prad"]
    teffkoi=observed_data["koi_steff"]
    mask=observed_data["koi_pdisposition"]=="CANDIDATE"
    mask=mask&(Pkoi>Pmin)&(Pkoi<Pmax)&(Rpkoi>Rpmin)&(Rpkoi<Rpmax)
    mask=mask&(teffkoi>4000)&(teffkoi<7000)

    
    Yobs=len(observed_data[mask])

    Ysum = np.float32(Yobs)
    abc.Ysm = np.array([Ysum])

#    print(Ysum)
#    sys.exit()
    #set prior parameters

    initep=Ysum*1.e-4

    
    # input model/prior

    ## input aux from the stellar catalog
    planet_data=pd.read_csv("data/kepler_berger.csv")
    rstar=planet_data["radiusnew"].values
    mstar=planet_data["mass"].values
    sigCDPP=planet_data["rrmscdpp04p5"].values
    mesthre=planet_data["mesthres04p5"].values
    Tdur = planet_data["dataspan"].values
    fduty = planet_data["dutycycle"].values
    
    mask=(sigCDPP==sigCDPP)&(mstar>0.0)&(rstar>0.0)&(mesthre==mesthre)&(Tdur==Tdur)&(fduty==fduty)

    #SELECT Main-Sequence
    teff=planet_data["teffnew"].values
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
    
    logPmin=np.log(Pmin)
    logPmax=np.log(Pmax)
    logRpmin=np.log(Rpmin)
    logRpmax=np.log(Rpmax)

    #switching condition of approx of the binomial distribution (see Binomial.ipynb)
    pchange=0.01458*(nstar/100000)**(-0.3333)
    
    #pre constraint of |cosi|
    RM3=2.0 #effective max of (Rstar/Mstar**1/3)
    RSOLAU=0.00464912633
    safefac=5.0 # safe factor ~ 1/(1-emax) emax=0.8
    pcrit=safefac*RSOLAU*RM3*(Pmin/365.242189)**(-2.0/3.0)

    abc.nparam=1
    abc.ntcommon=1 #use 1 thread common value in shared memory for Npick
    abc.model=\
    "#define Nstars "+str(nstar)+"\n"\
    +"#define PCRIT "+str(pcrit)+"\n"\
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

    abc.epsilon_list = np.array([initep,initep*0.9,initep*0.8])

    #initial run of abc pmc
    abc.check_preparation()
    abc.run()
    abc.check()

#    print(abc.dist)
#    print(abc.ntry)
#    print(abc.x)
    
#    fig=plt.figure(figsize=(10,5))
#    ax=fig.add_subplot(111)
#    plt.hist(abc.x,bins=300,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
#    plt.show()
#    sys.exit()
    #pmc sequence
    for eps in abc.epsilon_list[1:]:
        abc.run()
        abc.check()
        print("Mean distance=",np.mean(abc.dist))
    tend = time.time()

    print(tend-tstart,"sec")

    if True:
        fig=plt.figure(figsize=(10,5))
        ax=fig.add_subplot(111)
        plt.hist(abc.x,bins=30,label="$\epsilon$="+str(abc.epsilon),density=True,alpha=0.5)
        plt.title("Rp="+str(Rpmin)+"-"+str(Rpmax)+"Re, P="+str(Pmin)+"-"+str(Pmax)+"d")
        plt.xlabel("$f_{r,p}$")
        plt.ylabel("freqeuncy")
        plt.savefig("frp"+str(Rpmin)+"-"+str(Rpmax)+"_"+str(Pmin)+"-"+str(Pmax)+".png")
    
    return abc.x
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='compute frp')
    parser.add_argument('-p', nargs=2, help='Pmin, Pmax', type=float)
    parser.add_argument('-r', nargs=2, help='Rpmin, Rpmax', type=float)

    args = parser.parse_args()
    Pmin=args.p[0]
    Pmax=args.p[1]
    Rpmin=args.r[0]
    Rpmax=args.r[1]
    
    frp=ABCfrp(Pmin=Pmin,Pmax=Pmax,Rpmin=Rpmin,Rpmax=Rpmax)
    frpmed=np.median(frp)
    frpmin=np.percentile(frp,15.87)
    frpmax=np.percentile(frp,84.13)

    f = open('frp.txt','a')
    f.write(str(Rpmin)+","+str(Rpmax)+","+str(Pmin)+","+str(Pmax)+","+str(frpmed)+","+str(frpmin)+","+str(frpmax)+'\n')
    f.close()
