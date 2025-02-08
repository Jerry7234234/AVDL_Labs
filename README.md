# AVDL Labs

A brief description of the project.

## Table of Contents
- [Lab4](#Lab4)

## Lab1: Quantization and pruning
### Task 1
![Screenshot1](https://github.com/Jerry7234234/AVDL_Labs/blob/main/quantization.png)

### Task 2
![Screenshot2](https://github.com/Jerry7234234/AVDL_Labs/blob/main/pruning%201.png)

![Screenshot3](https://github.com/Jerry7234234/AVDL_Labs/blob/main/pruning%202.png)

## Lab2: Nerual architecture search
### Task 1
![Screenshot4](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20112836.png)

### Task 2
![Screenshot5](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20112554.png)

During the neural architectue search, we trained the model for 1 epoch. After compression, we train the compressed model for 2 epoches to fully recover the performance while maximizing the accuracy.

## Lab3: Mixed precision search
![Screenshot6](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20121256.png)

![Screenshot7](https://github.com/Jerry7234234/AVDL_Labs/blob/main/Screenshot%202025-02-04%20121516.png)

## Lab4
### Task 1
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
MXINT8 representation consists of many mantissa terms but only 1 exponent term. This means during an operation, the exponent part is loaded once but the mantissa part is loaded n times where n is the group size. Hence to implement MXINT8 opeartion naively in Python code, one will need to iterate the mantissa parts n times, which is not efficient. However, by using custom hardware that supports n bit parallel loading and can simutaneously load all the mantissa bits at once, this would greatly improve computation time by n times.

### b) What is the purpose of the variable dont_need_abs and bias in the C++ for loop?
During the conversion from MXINT8 to bfloat16, only the last 6 bits of the mantissas are used in the fraction part of the output. The 7th bit is ignored and the extracted 6 bits are left shifted by 1 to increase the bit length to 7. This can cause dequantization errors, as the 7th bit of the mantissa for MXINT8 numbers can represent either 1. or 0., but for bfloat16 the 7th bit is always 1. (In other words, bfloat16 and MXINT8 numbers have different dynamic range). For instance:


MXINT8: 01111111 | 00.111100 = $2 ^ {2 ^ 7 - 1 - 127} \times (2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 0.9375 = 0.9375$

Output bfloat16 number (without bias subtraction): 0 | 01111111 | 1111000 = $2 ^ {2 ^ 7 - 1 - 127} \times (1 + 2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 1.9375 = 1.9375$

As we can see, the fact the bfloat16 representation assumes a leading 1. being added to the fractional value leads to dequantization error. Hence by subtracting the bias:

output - bias = 0 | 01111111 | 1111000 - 0 | 01111111 | 0000000 =

(0 | 00000000 | 1111000 - 0 | 00000000 | 1000000) << 1 + 0 | 01111111 | 0000000 - 0 | 00000001 | 0000000 = 0 | 01111110 | 1110000 = 0.9375

The computed value now matches the bfloat16 value. In fact, for all MXINT8 mantissa that is of the form X0XXXXXX, the dequantization error will be present and the bias subtraction will need to be done.

### c i) How does cta_tiler partition data for copying to shared memory in CUDA kernel?
Tiling refers to subdivide and regrouping a multidimensional array of memeory for more efficient access. It uses the concept of layout, which are mappings from 2D coordinates to 1D indices of a memory space. This is useful since we are dealling with data in the form of high dimensional tensors and manipulation of each element in the tensor that were stored in specific locations of the memory can become inefficient if they are all pointed to difference indices. Tiling organises memory by dividing the 1D memory space into groups of multidimensional spaces that are more easily accessed during the loading of data from and to the shared memory. It can either divide the memeory space into intersecting groups with different strides, or into adjacent blocks with a stride of 1. The indices within each tile are partitioned by the input layout, which can be pre-defined by the programmer using layout composition. This means that when subdividing the data tensor into separate blocks of thread to be executed in parallel, each block will have access to the tile of memory that is the most optimal in terms of speed and memory efficiency.

### c ii) How does layout_sX partition threads in a threadblock for computation? 
Layout partition memeroy spaces so that data loading and computation can be more efficient. The cutlass layout uses composition and division to group memeory spaces into tiles. Each tile is then assigned to a thread based on the layout using the `local_partition()` function. The input data `x` is first subdivided into group tiles `mX` and then a shared memory is allocated to the group tiles to form `smem` and `sX`. The thread partition uses the group tiles `gX` and shared memory space `sX` together with layout to enable parallel computation with maximum memeory efficiency.

### Why the saved GPU memory is not exactly (32 - (4+8/32))/32 = 86.7%?
Firstly, the formula used to calculate memory efficiency is for MXINT4. Namely MXINT format with only 4 bits in the mantissa part. The MXINT format used in the experiment is MXINT8 (8 bits in the mantissa part), so the effective bandwidth should be (8+8/32). The correct saved memory prediction should be: (32 - (8+8/32))/32 = 74.2%. However, this is only the theoretical gain in memory efficiency. In actual runtime, there will be additional memory alllocation such as the shared memory and the pre-allocated global memory for the MXINT8 quantizer's tiling process, hence the actual saved memory is slightly less then this theoretical value.
