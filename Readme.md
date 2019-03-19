<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/logo.png" Titie="explanation" Width=150px>

GPU Approximate Bayesian Computation. Python and python-wrapped CUDA (using pycuda) code. Under development.

# Requirements and Setting

## Hardware

- Computer with NVIDIA GPU

Our test environments (GPU/OS) are GTX1070/linux mint19, PASCAL TITAN X /ubuntu18, GeForce TITAN X/linux mint19, GTX 1080Ti/ubuntu18, TESLA V100/ubuntu18. 

## Software

- python3
- cuda (cuda10 for our tests)
- pycuda
- numpy, scipy, matplotlib

**ABCFAST** uses header files (cuda kernels) for nvcc. set CPLUS_INCLUDE_PATH to abcfast/include.


For c-chell environment, write

```
 setenv CPLUS_INCLUDE_PATH abcfast/include
```

in .cshrc or .tchrc, so on. For bash environment, write

```
 CPLUS_INCLUDE_PATH=abcfast/include; export $CPLUS_INCLUDE_PATH
```

in .bashrc. Perform the **source** command.

# Examples

## ABC rejection sampling algorithm

- gabcrm_bin.py the GPU version of ABC rejection sampling algorithm, demonstrating a binomial example in Section 4 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub)

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm10.png" Titie="explanation" Width=250px><img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm100.png" Titie="explanation" Width=250px><img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm1000.png" Titie="explanation" Width=250px>

ABC posterior (histogram) and analytic solutions (solid) for the binomial example (n=10,100, and 1000), using N=10000 particles.

## ABC-PMC algorithm

- gabcpmc_exp.py the GPU version of ABC PMC algorithm, demonstrating an exponential example in Section 5 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub), with some modifications.

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcpmc.png" Titie="explanation" Width=450px>

ABC posteriors for different tolerance thresholds. The summary statistics is s=|mean(X) - mean(Y)|.

- gabcpmc_norm2d.py demonstrating a 2D gaussian case, inspired from 2d_gauss example in [abcpmc](https://github.com/jakeret/abcpmc) package by Akeret et al. 

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcpmc_norm2d.png" Titie="explanation" Width=750px>

- gabcpmc_sumnorm.py demonstrating a gaussian+gaussian case (Beaumont+09). 

## Customizing the ABC-PMC

Prepare the following functions:

### model
data sampler with given (model) parameters,

**Ysim(NDATA) ~ model( param(NPARAM) )**,

cuda-based or prepared python function

### prior
parameter sample from a prior distribution,

**param(NPARAM) ~ prior()**, 

cuda-based or prepared python function  

### fprior
a prior density distribution to compute weights, python function.


## Hierarchical ABC-PMC

- gabcpmc_hibin.py  demonstrating an exponential example in Section 6 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub). Still unstable though.

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/hibin.png" Titie="explanation" Width=750px>

The summary statistics is s = Sum (for subject) |mean(X) - mean(Y)|

## Customizing the Hierarchical ABC-PMC

Prepare

### model
data sampler with given (model) parameters,

**Ysim(NDATA) ~ model( param(NPARAM) )**,

cuda-based or prepared python function

### prior
parameter sampler from a prior distribution,

**param(NPARAM) ~ prior( hparam(NHPARAM) )**, 

cuda-based or prepared python function  

### hyperprior
hyperparameter sampler from a hyperprior distribution,

**hparam(NHPARAM) ~ hyperprior()**, 

cuda-based or prepared python function  


### fhprior
a hyperprior density distribution to compute weights, python function

## Random number generators using curand_kernel.h

Directory: random_gen

- uniform, normal, 2D normal, gamma distribution, beta distribution, binomial distribution, exponential distribution, random choise with discrete probability p_i (the alias method).

# Current Status

This code is in the beta stage (very unstable). Collaboration with risk sharing or feedback is welcome. Ask [Hajime Kawahara](http://secondearths.sakura.ne.jp/en/index.html) for more details.

