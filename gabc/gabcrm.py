from pycuda.compiler import SourceModule

def gabcrm_module ():
    source_module = SourceModule("""

    #include <stdio.h>
    #define MAXTRY 100000
    
    __device__ int rho(int X,int Y){

    return abs(X - Y);

    }

    #include "abcrm.h"

    """,options=['-use_fast_math'],no_extern_c=True)

    return source_module
