GPU Approximate Baysian Computation

Under development.

# setting

gabc uses header files for nvcc. set CPLUS_INCLUDE_PATH to gabc/include.

```
 setenv CPLUS_INCLUDE_PATH /../../exoabc/include

```



# ABC samples

## ABC rejection sampling algorithm

- gabcrm.py the GPU version of ABC rejection sampling algorithm, demonstrating a binomial example in Section 4 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub)

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm10.png" Titie="explanation" Width=250px><img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm100.png" Titie="explanation" Width=250px><img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm1000.png" Titie="explanation" Width=250px>

ABC posterior (histogram) and analytic solutions (solid) for the binomial example (n=10,100, and 1000), using N=10000 particles.

## ABC PMC algorithm

- gabcpmc_exp.py the GPU version of ABC PMC algorithm, demonstrating an exponential example in Section 5 of [Turner and Van Zandt (2012) JMP 56, 69](https://www.sciencedirect.com/science/article/abs/pii/S0022249612000272?via%3Dihub), with some modifications.

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/pmc_exp_hh.png" Titie="explanation" Width=450px>

ABC posteriors for different tolerance thresholds. The summary statistics is s=|mean(X) - mean(Y)|.

| tolerance | Time sec (GTX1070) | Time sec (Tesla V100) |
|:----------|-----------:|:------------:|
| 3.0  | 0.29 | 0.20 |
| 1.0  | 0.11 | 0.01 |
| 0.1  | 0.12 | 0.01 |
| 1e-3 | 0.25 | 0.03 |
| 1e-4 | 1.4  | 0.16 |
| 1e-5 | 14.1 | 1.3 |
| total| 16.4 | 1.7 |

Note that the unit of time is not minutes but second.

# Random number generators using curand_kernel.h

Directory: random_gen

- uniform, normal, gamma distribution, beta distribution, binomial distribution, random choise with discrete probability p_i (the alias method).

