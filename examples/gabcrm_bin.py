from abcfast.gabcrm import *
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    import time
    from scipy.stats import beta as betafunc

    tstart=time.time()
    
    print("*******************************************")
    print("GPU ABC Rejection Method.")
    print("This code demonstrates a binomial example in Section 4 in Turner and Van Zandt (2012) JMP 56, 69")
    print("*******************************************")
    n=10
    ptrue=0.7
    Yobs=random.binomial(n,ptrue)
    epsilon = 1
    print("Observed Value is ",Yobs)
    
    nw=1
    nt=10000
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte
    x=np.zeros(nb)
    x=x.astype(np.float32)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)
    
    source_module=gabcrm_module()
    pkernel=source_module.get_function("abcrm")
    pkernel(dev_x,np.int32(Yobs),np.int32(n),np.int32(epsilon),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
    cuda.memcpy_dtoh(x, dev_x)

    tend=time.time()
    print("t=",tend-tstart)
    
    plt.hist(x,bins=30,label="n="+str(n),density=True,alpha=0.5)
    plt.axvline(ptrue,color="gray",label="True",ls="dashed")
    plt.xlim(0,1)
    plt.xlabel("p",fontsize=16)
    alpha=1.0
    beta=1.0
    
    xl = np.linspace(betafunc.ppf(0.0001, Yobs+alpha, n - Yobs + beta),betafunc.ppf(0.9999,Yobs+alpha, n - Yobs + beta), 100)
    plt.plot(xl, betafunc.pdf(xl, Yobs+alpha, n - Yobs + beta),label="analytic", color="green")
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.title("Observed = "+str(Yobs),fontsize=14)
    plt.savefig("abcrm"+str(n)+".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
