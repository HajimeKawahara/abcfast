GPU Approximate Baysian Computation

Under development.

# ABC samples

## ABC rejection sampling

- gabcrm.py the GPU version of ABC rejection sampling algorithm, demonstrating a binomial example in Section 4 in Turner and Van Zandt (2012) JMP 56, 69

<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm10.png" Titie="explanation" Width=250px>
<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm100.png" Titie="explanation" Width=250px>
<img src="https://github.com/HajimeKawahara/gabc/blob/master/documents/fig/abcrm1000.png" Titie="explanation" Width=250px>

ABC posterior (histogram) and analytic solutions (solid) for the binomial example (n=10,100, and 1000), using N=10000 particles.


# Random number generators

Directory: random_gen

- uniform, normal, gamma distribution, beta distribution, binomial distribution
