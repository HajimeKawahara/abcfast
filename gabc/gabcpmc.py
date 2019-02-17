import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
from gabc.utils.statutils import *
import sys

def gabcpmc_module (model,prior):
    header=\
    """
    #include <stdio.h>
    #include <math.h>
    #include <curand_kernel.h>
    #include "gengamma.h"
    #define MAXTRYX 10000000

    extern __shared__ volatile float cache[]; 

    """
    footer=\
    """
    #include "abcpmc_init.h"
    #include "abcpmc.h"
    #include "compute_weight.h"

    """

    source_module = SourceModule(header+model+prior+footer,options=['-use_fast_math'],no_extern_c=True)

    return source_module


def setmem_device(npart,dtype):
    x=np.ones(npart)
    x=x.astype(dtype)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)
    return x,dev_x

    
class ABCpmc(object):
    def __init__(self,npart,Yobs):
        self.npart = npart
        self.Yobs = Yobs
        self.n = len(Yobs)
        if checkpower2(npart):
            print("npart=",npart)
            sys.exit("Error: Use power of 2 as npart (# of the particles).")

        self.ptwo = getptwo(self.n)


        self.wide=8.0
        self.epsilon_list = False
        self.nthread_use_max=512 # maximun number of the threads in a block for use

        self.x,self.dev_x=setmem_device(self.npart,np.float32)
        self.xx,self.dev_xx=setmem_device(self.npart,np.float32)

        self.ntry,self.dev_ntry=setmem_device(self.npart,np.int32)
        self.dist,self.dev_dist=setmem_device(self.npart,np.float32)

        self.iteration = 0
        self.epsilon = None
        self._model = None
        self._prior = None
        self._parprior = None
        self._dev_parprior = None
        self.sigmat_prev = None
        self.dev_Ki = None
        self.dev_Li = None
        self.dev_Ui = None
        self.seed = -1

        

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self,model):
        self._model = model
        self.update_kernel()

    @property
    def prior(self):
        return self._prior
    
    @prior.setter
    def prior(self,prior):
        self._prior = prior
        self.update_kernel()

    @property
    def parprior(self):
        return self._parprior

    @parprior.setter
    def parprior(self,parprior):
        self._parprior = parprior.astype(np.float32)
        self.dev_parprior = cuda.mem_alloc(self._parprior.nbytes)
        cuda.memcpy_htod(self.dev_parprior,self._parprior)
        
    def update_kernel(self):        
        if self._model is not None and self._prior is not None:
            self.source_module=gabcpmc_module(self._model,self._prior)
            self.pkernel_init=self.source_module.get_function("abcpmc_init")
            self.pkernel=self.source_module.get_function("abcpmc")
            self.wkernel=self.source_module.get_function("compute_weight")
    
    def run(self):

        if self.iteration == 0:

            print("currently |X-Y|/n is only available for summary statistics.")
            Ysum=np.sum(self.Yobs)
            self.epsilon=self.epsilon_list[self.iteration]
            sharedsize=(self.n+1)*4 #byte
            self.pkernel_init(self.dev_x,np.float32(Ysum),np.float32(self.epsilon),np.int32(self.seed),self.dev_parprior,self.dev_dist,self.dev_ntry,np.int32(self.ptwo),block=(int(self.n),1,1), grid=(int(self.npart),1),shared=sharedsize)

            cuda.memcpy_dtoh(self.x, self.dev_x)
            cuda.memcpy_dtoh(self.ntry, self.dev_ntry)

            self.sigmat_prev = np.sqrt(self.wide*np.var(self.x))
            self.iteration = 1

            self.init_weight()
            
        else:

            print("currently |X-Y|/n is only available for summary statistics.")
            Ysum=np.sum(self.Yobs)
            self.epsilon=self.epsilon_list[self.iteration]
            sharedsize=(self.n+1)*4 #byte

            self.pkernel(self.dev_xx,self.dev_x,np.float32(Ysum),np.float32(self.epsilon),self.dev_Ki,self.dev_Li,self.dev_Ui,np.float32(self.sigmat_prev),np.int32(self.seed),self.dev_dist,self.dev_ntry,np.int32(self.ptwo),block=(int(self.n),1,1), grid=(int(self.npart),1),shared=sharedsize)

            cuda.memcpy_dtoh(self.x, self.dev_xx)
            cuda.memcpy_dtoh(self.ntry, self.dev_ntry)
            self.sigmat_prev = np.sqrt(self.wide*np.var(self.x))
            #update weight
            self.update_weight()
            #swap
            self.dev_x, self.dev_xx = self.dev_xx, self.dev_x
            self.dev_w, self.dev_ww = self.dev_ww, self.dev_w
            self.iteration = self.iteration + 1
            
    def check(self):
        FR=len(self.x[self.x<0])/len(self.x)
        print("#"+str(self.iteration-1)+":","epsilon=",self.epsilon,"Fail Rate=",FR)
        if FR>0:
            print("ERROR: Increase epsilon or MAXVALX in kernel.")
            sys.exit("")
        print("mean max min = ",np.mean(self.ntry),np.max(self.ntry),np.min(self.ntry))

    def init_weight(self):
        #window
        self.w,self.dev_w=setmem_device(self.npart,np.float32)
        self.ww,self.dev_ww=setmem_device(self.npart,np.float32)
        
        Ki,Li,Ui=genalias_init(self.w)
        self.dev_Ki = cuda.mem_alloc(Ki.nbytes)
        self.dev_Li = cuda.mem_alloc(Li.nbytes)
        self.dev_Ui = cuda.mem_alloc(Ui.nbytes)        
        cuda.memcpy_htod(self.dev_Ki,Ki)
        cuda.memcpy_htod(self.dev_Li,Li)
        cuda.memcpy_htod(self.dev_Ui,Ui)

    def update_weight(self):
        #update weight
        sharedsize=int(self.npart*4) #byte
        nthread=min(self.npart,self.nthread_use_max)
        
        self.wkernel(self.dev_ww, self.dev_w, self.dev_xx, self.dev_x, np.float32(self.sigmat_prev), block=(int(nthread),1,1), grid=(int(self.npart),1),shared=sharedsize)

        cuda.memcpy_dtoh(self.w, self.dev_ww)
        #
        pri=self.fprior(self.x, self.parprior)
        #
        self.w=pri/self.w
        self.w=self.w/np.sum(self.w)
        self.w=self.w.astype(np.float32)
        Ki,Li,Ui=genalias_init(self.w)
        cuda.memcpy_htod(self.dev_Ki,Ki)
        cuda.memcpy_htod(self.dev_Li,Li)
        cuda.memcpy_htod(self.dev_Ui,Ui)


    