* model preparation

** Using global memory in model

- Set abc.aux.
- Use aux[] in model (aux arrays).
- isample specifies sample index. For instance, if you use the aux array for each sample (with aux[0:Nsample-1]), each thread may call aux[isample].

** Using shared memory in model

- Set abc.ntcommon (the number of thread common values).
- In the shared memory, cache[0:NRESERVED-1] are reserved. So, use cache[NRESERVED:NRESERVED+(ntcommon)].
