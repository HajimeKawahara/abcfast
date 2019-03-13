<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/logo.png" Titie="explanation" Width=150px>

GPU Approximate Bayesian Computation. Under development.

# setting

**ABCFAST** uses header files for nvcc. set CPLUS_INCLUDE_PATH to gabc/include.

```
 setenv CPLUS_INCLUDE_PATH abcfast/include
```

# Examples

## ABC rejection sampling algorithm

- gabcrm_bin.py the GPU version of ABC rejection sampling algorithm, demonstrating a binomial example in Section 4 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub)

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm10.png" Titie="explanation" Width=250px><img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm100.png" Titie="explanation" Width=250px><img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm1000.png" Titie="explanation" Width=250px>

ABC posterior (histogram) and analytic solutions (solid) for the binomial example (n=10,100, and 1000), using N=10000 particles.

## ABC PMC algorithm

- gabcpmc_exp.py the GPU version of ABC PMC algorithm, demonstrating an exponential example in Section 5 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub), with some modifications.

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcpmc.png" Titie="explanation" Width=450px>

ABC posteriors for different tolerance thresholds. The summary statistics is s=|mean(X) - mean(Y)|.

- gabcpmc_norm2d.py demonstrating a 2D gaussian case, inspired from 2d_gauss example in [abcpmc](https://github.com/jakeret/abcpmc) package by Akeret et al. 

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcpmc_norm2d.png" Titie="explanation" Width=750px>

- gabcpmc_sumnorm.py demonstrating a gaussian+gaussian case (Beaumont+09). 

# Customizing ABC-PMC

You need to prepare

### model
data sampler with given (model) parameters,

**Ysim(NDATA) ~ model( param(NPARAM) )**,

c based or prepared python function

### prior
parameter sample from a prior distribution,

**param(NPARAM) ~ prior()**, 

c based or prepared python function  

### fprior
a prior density distribution to compute weights, python function


# Hierarchical ABC

- gabcpmc_hibin.py  demonstrating an exponential example in Section 6 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub). Still super unstable though.

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/hibin.png" Titie="explanation" Width=750px>



# Random number generators using curand_kernel.h

Directory: random_gen

- uniform, normal, 2D normal, gamma distribution, beta distribution, binomial distribution, random choise with discrete probability p_i (the alias method).


# Current Status

This code is in the beta stage (very unstable). Ask [Hajime Kawahara](http://secondearths.sakura.ne.jp/en/index.html) for more details.

