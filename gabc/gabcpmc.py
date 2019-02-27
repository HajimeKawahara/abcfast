import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
from gabc.utils.statutils import *
import sys

def gabcpmc_module (model,prior,nmodel,ndata,nsample,maxtryx=10000000):
    header=\
    "    #define NMODEL "+str(nmodel)+"\n"\
    +"    #define NDATA "+str(ndata)+"\n"\
    +"    #define NSAMPLE "+str(nsample)+"\n"\
    +"    #define MAXTRYX "+str(maxtryx)+"\n"\
    +"""
    #define CUDART_NAN_F __int_as_float(0x7fffffff)
    #include <stdio.h>
    #include <math.h>
    #include <curand_kernel.h>

    extern __shared__ volatile float cache[]; 

    """\
    
    footer=\
    """
    #include "abcpmc_init.h"
    #include "abcpmc.h"
    #include "compute_weight.h"

    """
    print(header+model+prior+footer)
    source_module = SourceModule(header+model+prior+footer,options=['-use_fast_math'],no_extern_c=True)

    return source_module


def setmem_device(npart,dtype):
    x=np.ones(npart)
    x=x.astype(dtype)
    dev_x = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod(dev_x,x)
    return x,dev_x

    
class ABCpmc(object):
    def __init__(self):

        self.maxtryx = 10000000 #MAXTRYX reduce this value when you debug the code.
        self.nthread_max = 1024 #MAX NUMBER OF THREADS IN A BLOCK

        self._npart = 512  # number of the particles (default=512)
        self._nmodel = None    # number of the model parameters 
        self._nsample = None # number of the data vector
        self._ndata = None # dimension of the data vector
        self._Ysm = None # summary statistics vector        
        self.nthread = None #number of threads    

        #        self.nsm = None # dimension of the summary statistics vector
        
        self.wide=8.0
        self.epsilon_list = False
        self.nthread_use_max=512 # maximun number of the threads in a block for use

        self.x=None
        self.xw = None #(npart,nmodel)-dimension array type of x
        self.dev_x=None
        self.xx=None
        self.dev_xx=None
        self.ntry=None
        self.dev_ntry=None
        self.dist=None
        self.dev_dist=None
        self.invcov = None
        self.dev_invcov = None
        self.Qmat = None
        self.dev_Qmat = None
        
        self.iteration = 0
        self.epsilon = None
        self._model = None
        self._prior = None
        self._parprior = None
        self._dev_parprior = None

        self.dev_Ki = None
        self.dev_Li = None
        self.dev_Ui = None
        self.dev_Ysm = None

        self.seed = -1
        self.prepare = False

    @property
    def ndata(self):
        return self._ndata

    @ndata.setter
    def ndata(self,ndata):
        self._ndata = ndata
        self.update_kernel()

    @property
    def nsample(self):
        return self._nsample

    @nsample.setter
    def nsample(self,nsample):
        self._nsample = nsample
        self.ptwo = getptwo(self._nsample)
        self.update_kernel()

        
    @property
    def nmodel(self):
        return self._nmodel
    
    @nmodel.setter
    def nmodel(self,nmodel):
        self._nmodel = nmodel
        self.update_kernel()
        
    @property
    def npart(self):
        return self._npart

    @npart.setter
    def npart(self,npart):
        if checkpower2(npart):
            print("npart(icles)=",npart)
            sys.exit("Error: Use power of 2 as npart (# of the particles).")
        self._npart = npart
        self.update_kernel()
        
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
        if self._model is not None \
           and self._prior is not None and self._npart is not None \
           and self._nmodel is not None and self._ndata is not None \
           and self._nsample is not None:
            
            self.source_module=gabcpmc_module(self._model,self._prior,self._nmodel,self._ndata,self._nsample,maxtryx=self.maxtryx)
            self.pkernel_init=self.source_module.get_function("abcpmc_init")
            self.pkernel=self.source_module.get_function("abcpmc")
            self.wkernel=self.source_module.get_function("compute_weight")
            
            self.x,self.dev_x=setmem_device(self._npart*self._nmodel,np.float32)
            self.xx,self.dev_xx=setmem_device(self._npart*self._nmodel,np.float32)
            self.ntry,self.dev_ntry=setmem_device(self._npart,np.int32)
            self.dist,self.dev_dist=setmem_device(self._npart,np.float32)
            self.invcov,self.dev_invcov=setmem_device(self._nmodel*self._nmodel,np.float32)
            self.Qmat,self.dev_Qmat=setmem_device(self._nmodel*self._nmodel,np.float32)
            self.nthread = min(self.nthread_max,self._nsample)
            self.prepare = True
            
    @property
    def Ysm(self):
        return self._Ysm
    
    @Ysm.setter
    def Ysm(self,Ysm):
        self._Ysm = Ysm.astype(np.float32)
        self.dev_Ysm = cuda.mem_alloc(self._Ysm.nbytes)        
        cuda.memcpy_htod(self.dev_Ysm,self._Ysm)
#        self.nsm = len(self._Ysm)
        
    def run(self):
        
        if self.iteration == 0:

            self.epsilon=self.epsilon_list[self.iteration]
            sharedsize=(self._nsample*self._ndata+self._nmodel)*4 #byte
            self.pkernel_init(self.dev_x,self.dev_Ysm,np.float32(self.epsilon),np.int32(self.seed),self.dev_parprior,self.dev_dist,self.dev_ntry,np.int32(self.ptwo),block=(int(self.nthread),1,1), grid=(int(self._npart),1),shared=sharedsize)

            cuda.memcpy_dtoh(self.x, self.dev_x)
            
            #update covariance
            self.update_invcov()
            #update weight
            self.init_weight()
            self.iteration = 1
            
        else:

            self.epsilon=self.epsilon_list[self.iteration]
            sharedsize=(self._nsample*self._ndata+self._nmodel)*4 #byte
            self.pkernel(self.dev_xx,self.dev_x,self.dev_Ysm,np.float32(self.epsilon),self.dev_Ki,self.dev_Li,self.dev_Ui,self.dev_Qmat,np.int32(self.seed),self.dev_dist,self.dev_ntry,np.int32(self.ptwo),block=(int(self.nthread),1,1), grid=(int(self._npart),1),shared=sharedsize)

            cuda.memcpy_dtoh(self.x, self.dev_xx)
            
            #update covariance
            self.update_invcov()
            #update weight
            self.update_weight()
            #swap
            self.dev_x, self.dev_xx = self.dev_xx, self.dev_x
            self.dev_w, self.dev_ww = self.dev_ww, self.dev_w
            self.iteration = self.iteration + 1
            
    def check(self):
        cuda.memcpy_dtoh(self.ntry, self.dev_ntry)
        FR=len(self.x[self.x!=self.x])/len(self.x)
        print("#"+str(self.iteration-1)+":","epsilon=",self.epsilon,"Fail Rate=",FR)
        if FR>0:
            print("ERROR: Increase epsilon or MAXVALX in kernel.")
            sys.exit("")
        print("mean max min = ",np.mean(self.ntry),np.max(self.ntry),np.min(self.ntry))

    def init_weight(self):
        #window
        self.w,self.dev_w=setmem_device(self._npart,np.float32)
        self.ww,self.dev_ww=setmem_device(self._npart,np.float32)
        
        Ki,Li,Ui=genalias_init(self.w)
        self.dev_Ki = cuda.mem_alloc(Ki.nbytes)
        self.dev_Li = cuda.mem_alloc(Li.nbytes)
        self.dev_Ui = cuda.mem_alloc(Ui.nbytes)        
        cuda.memcpy_htod(self.dev_Ki,Ki)
        cuda.memcpy_htod(self.dev_Li,Li)
        cuda.memcpy_htod(self.dev_Ui,Ui)

    def update_invcov(self):
        
        #inverse covariance matrix
        if self._nmodel == 1:
            cov = self.wide*np.var(self.x)
            self.invcov = np.array(1.0/cov).astype(np.float32)
            self.Qmat = np.array([np.sqrt(cov)]).astype(np.float32)
        else:
            self.xw=np.copy(self.x).reshape(self._npart,self._nmodel)
            cov = self.wide*np.cov(self.xw.transpose(),bias=True)
            self.invcov = (np.linalg.inv(cov).flatten()).astype(np.float32)
            # Q matrix for multivariate Gaussian prior sampler
            [eigenvalues, eigenvectors] = np.linalg.eig(cov)
            l = np.matrix(np.diag(np.sqrt(eigenvalues)))
            Q = np.matrix(eigenvectors) * l
            self.Qmat=(Q.flatten()).astype(np.float32)

        cuda.memcpy_htod(self.dev_invcov,self.invcov)
        cuda.memcpy_htod(self.dev_Qmat,self.Qmat)
        
        
    def update_weight(self):
        #update weight
        sharedsize=int(self._npart*4) #byte
        nthread=min(self._npart,self.nthread_use_max)
        
        self.wkernel(self.dev_ww, self.dev_w, self.dev_xx, self.dev_x, self.dev_invcov, block=(int(nthread),1,1), grid=(int(self._npart),1),shared=sharedsize)

        cuda.memcpy_dtoh(self.w, self.dev_ww)

        if self._nmodel == 1:            
            pri=self.fprior(self.x, self.parprior)
        else:
            pri=self.fprior(self.xw, self.parprior)
    
        self.w=pri/self.w
        self.w=self.w/np.sum(self.w)
        self.w=self.w.astype(np.float32)
        Ki,Li,Ui=genalias_init(self.w)
        cuda.memcpy_htod(self.dev_Ki,Ki)
        cuda.memcpy_htod(self.dev_Li,Li)
        cuda.memcpy_htod(self.dev_Ui,Ui)


    def check_preparation(self):
        if not self.prepare:
            print("Error: parameter setting is imcomplete:")
            if self._model is None:
                print("SET .model")
            if self._prior is None:
                print("SET .prior")
            if self._npart is None:
                print("SET .npart (# of the particles)")
            if self._nmodel is None:
                print("SET .nmodel (# of the model parameters)")
            if self._ndata is None:
                print("SET .ndata (# of the output parameters of the model)")
            if self._nsample is None:
                print("SET .nsample (# of the samples)")            
            sys.exit()

        if self._Ysm is None:
            print("SET .Ysm (summary statistics of data)")
            sys.exit()
            
