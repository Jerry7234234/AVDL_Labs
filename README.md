# AVDL Labs

A brief description of the project.

## Table of Contents
- [Lab4](#Lab4)

## Lab1: Quantization and pruning
### Task 1
![Screenshot1](https://github.com/Jerry7234234/AVDL_Labs/blob/main/quantization.png)

Post quantization training brought the accuracy back to the same level as the unquantizaed model.

### Task 2
![Screenshot2](https://github.com/Jerry7234234/AVDL_Labs/blob/main/NAS2.png)

We can see that structured pruning using L1-norm regulerisation achieves a higher maximum accuracy than random structured pruning. To better visualize the performance of the two methods, we can plot the actual accuracy per iteration. As shown below:

![Screenshot3](https://github.com/Jerry7234234/AVDL_Labs/blob/main/pruning%202.png)

The performance of L1-norm pruning is stable and consistent and accuracy only starts to drop at the highest sparisities. The performance of random pruning is noisy and the accuracy flattens to 0.5 after a certain sparsity threshold. This makes sense as L1-norm ranks weights based on their summed absolute values hence only the weights with the samllest influence will be removed, minimising the effect of pruning on accuracy. Whereas random pruning make accidentally removes important weights and adversely affects the accuracy.

## Lab2: Nerual architecture search
### Task 1
![Screenshot4](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20112836.png)

We can see that the TPE sampler achieves the highest accuracy. This is because TPE sampler extracts the best performing samples and uses kernel density estimation to generalize the best samples to a greater search space in an attempt to discover better solutions amoung the current best solutions. The process then repeats until it converges to the best possible solution. Hence the name "tree-structred". The grid sampler can in theory achieves the best accuracy among all. However, this is only true if we sample all the combinations of configurations and it doesn't perform well with limited trials, especially when the allowed search space range is much smaller than the complete range.

### Task 2
![Screenshot5](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20112554.png)

During the neural architectue search, we trained the model for 1 epoch. After compression, we train the compressed model for 2 epoches to fully recover the performance while maximizing the accuracy.

## Lab3: Mixed precision search
![Screenshot6](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20121256.png)

![Screenshot7](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20121516.png)

## Lab4
### Task 1 a)
We can see that the optimized model does not perfrom any better than the original model, and there are a few possible reasons: (1) The `torch.compile()` funciton introduces a delay caused by compilation overhead, which means the inference speed of the model during the initial iterations are slower and speeds up later on. The fact that we're only using n=5 means a propotion of the runtime operation were occupied by this delay. (2) The input data size is not particularly large, this makes the benefits of using a torch compiler less obvious as the model should perform equally well on small data regardless of whether it was compiled.

### Task 1 b)
Although there's no obvious pattern, the general trend is that the optimized model timed using cpu computes relatvely faster then that timed using cuda. This is likely because a cpu model benefits more from the compiler as it removes unneccesay computationt stpes and fused kernel to save up memory and loading time. On the other hand, gpu models are already optimized and has little improvement after compiling. Moreover, since resnet18 is not a large model, the effect of `torch.compile` is less obvious.

| Implementation  | CPU  | GPU |
| --------------- | ---- | --- |
|  naive   | 9.7964s | 0.1352s |
| compiled | 5.4800s | 0.1362s |

### Task 2
The same observation is obtained with the sacled dot produc attention test. For the timed test ran with cpu, the fused SDPA kernel execute significantly faster then the primitive implementation. However, when using cuda device computation, the difference in computation time is negligible.

| Implementation  | CPU  | GPU |
| --------------- | ---- | --- |
| naive | 0.8423s | 0.0007s |
| fused | 0.0243s | 0.0003s |

### Task 3
### a) How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?
MXINT8 format reduces quantization error as it can represent $2^8 * 2^8 = 65536$ values, reducing the workload for custom hardware as large bandwidth floating point operations are not needed. Also, since the exponent part is shared, this makes matrix multiplication operation much easier as it is only applied once at the end of the operation, which saves computation overhead and speeds up the calculation process. Moreover, since each mantissa part is only an 8 bit fixed point number, it can reduce the bandwidth to save up memory in hardware while also allows a simpler gpu architecture by requiring less logic units, these ultimately leads to less power consumption and memory bottleneck.

In terms of low level operation, MXINT8 consists of many mantissa terms but only 1 exponent term. This means during an operation, the exponent part is loaded once but the mantissa part is loaded n times where n is the group size. Hence to implement MXINT8 opeartion naively in Python code, one will need to iterate the mantissa parts n times, which is not efficient. However, by using custom hardware that supports n bit parallel loading and can simutaneously load all the mantissa bits at once, this would greatly improve computation time by n times.

### b) What is the purpose of the variable dont_need_abs and bias in the C++ for loop?
During the conversion from MXINT8 to bfloat16, only the last 6 bits of the mantissas are used in the fraction part of the output. The 7th bit is ignored and the extracted 6 bits are left shifted by 1 to increase the bit length to 7. This can cause dequantization errors, as the 7th bit of the mantissa for MXINT8 numbers can represent either 1. or 0., but for bfloat16 the 7th bit is always 1. (In other words, bfloat16 and MXINT8 numbers have different dynamic range). For instance:

MXINT8: 01111111 | 00.111100 = $2 ^ {2 ^ 7 - 1 - 127} \times (2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 0.9375 = 0.9375$

If we force that to be bfloat16 number (without bias subtraction) then: 0 | 01111111 | 1111000 = $2 ^ {2 ^ 7 - 1 - 127} \times (1 + 2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 1.9375 = 1.9375$

As we can see, the fact the bfloat16 representation assumes a leading 1. being added to the fractional value means that whenever the 7th bit of mantissa is 0, the converted results will have an error of 1.0, leading to dequantization error. Hence by subtracting the bias:

output - bias = 0 | 01111111 | 1111000 (bfloat16) - 0 | 01111111 | 0000000 (bfloat16) = 1.9375 - 1.0 =

(0 | 00000000 | 1111000 - 0 | 00000000 | 1000000) << 1 + 0 | 01111111 | 0000000 - 0 | 00000001 | 0000000 = 0 | 01111110 | 1110000 = 0.9375

The computed value now matches the bfloat16 value. In fact, for all MXINT8 mantissa that is of the form X0XXXXXX, the dequantization error will be present and the bias subtraction will need to be done.

### c i) How does cta_tiler partition data for copying to shared memory in CUDA kernel?
Tiling refers to subdivide and regrouping a multidimensional array of memeory for more efficient access. It uses the concept of layout, which are mappings from 2D coordinates to 1D indices of a memory space. This is useful since we are dealling with data in the form of high dimensional tensors and manipulation of each element in the tensor that were stored in specific locations of the memory can become inefficient if they are all pointed to difference indices.

Tiling organises memory by dividing the 1D global memory into groups of multidimensional spaces that are more easily accessed during the loading of data from and to the shared memory.

```
Tensor mX_raw = make_tensor(make_gmem_ptr(x_raw_int8), shape_x, stride_x);
Tensor mX = flatten(flat_divide(mX_raw, group_tiler)); // (_group_size, num_groups):(_1, _group_size)
```
```
Tensor gX = local_tile(mX, cta_tiler, cta_coord);                 
Tensor gScale = local_tile(vScale, select<1>(cta_tiler), select<1>(cta_coord));
Tensor gY = local_tile(mY, cta_tiler, cta_coord);
```

It can either divide the memeory space into intersecting groups with different strides, or into adjacent blocks with a stride of 1. The indices within each tile are partitioned by the input layout, which can be pre-defined by the programmer using layout composition. 

![Screenshot8](https://github.com/Jerry7234234/AVDL_Labs/blob/main/tiling.png)

This means that when subdividing the data tensor into separate blocks of thread to be executed in parallel, each block will have access to the tile of memory that is the most optimal in terms of speed and memory efficiency.

### c ii) How does layout_sX partition threads in a threadblock for computation? 
Layout partition memeroy spaces so that data loading and computation can be more efficient. The cutlass layout uses composition and division to group memeory spaces into tiles. Each tile is then assigned to a thread based on the layout using the `local_partition()` function. 

```
Tensor tXgX = local_partition(gX, layout_tX, threadIdx.x);
Tensor tXsX = local_partition(sX, layout_sX, threadIdx.x);
Tensor tXgY = local_partition(gY, layout_tX, threadIdx.x);
...
Tensor tXcX = local_partition(cX, layout_sX, threadIdx.x);
```

The input data `x` is first subdivided into group tiles `mX` and then a shared memory is allocated to the group tiles to form `smem` and `sX`. The thread partition uses the group tiles `gX` and shared memory space `sX` together with layout to enable parallel computation with maximum memeory efficiency.

Example of using layout in 2D tiling:
![Screenshot9](https://github.com/Jerry7234234/AVDL_Labs/blob/main/layout.png)

### Why the saved GPU memory is not exactly (32 - (4+8/32))/32 = 86.7%?
First obvious answer is that the formula used to calculate memory efficiency is for MXINT4. Namely MXINT format with only 4 bits in the mantissa part. The MXINT format used in the experiment is MXINT8 (8 bits in the mantissa part), so the effective bandwidth should be (8+8/32). The correct saved memory prediction should be: (32 - (8+8/32))/32 = 74.2%.

However, this is only the theoretical gain in memory efficiency. In actual runtime, there will be additional memory alllocation for the quantization kernel, such as the shared memory and the pre-allocated global memory for the MXINT8 quantizer's tiling process. Moreover, since we are not quantizing the entire model, some operations will still use FP32. For example, the activation or attention layers. Hence the actual saved memory is slightly less then this theoretical value. Some other more subtle reasons could be associated with the inefficient gpu caching which resulted in redundant memory. 
