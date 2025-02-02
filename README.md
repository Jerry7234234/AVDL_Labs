# AVDL Labs

A brief description of the project.

## Table of Contents
- [Lab4](#Lab4)

## Lab4
### Task 1 a)
Although there's no obvious pattern, the general trend is that the optimized model timed using cpu computes relatvely faster then that timed using cuda. This is likely because a cpu model benefits more from the compiler as it removes unneccesay computationt stpes and fused kernel to save up memory and loading time. On the other hand, gpu models are already optimized and has little improvement after compiling. Moreover, since resnet18 is not a large model, the effect of `torch.compile` is less obvious.
### Question: What is the purpose of the dont_need_abs variable and the bias variable? Note that unlike IEEE Floating-Point, MXINT has no implicit leading bit for the mantissa.
During the conversion from MXINT8 to bfloat16, only the last 6 bits of the mantissas are used in the fraction part of the output. The 7th bit is ignored and the extracted 6 bits are left shifted by 1 to increase the bit length to 7. This can cause dequantization errors, as the 7th bit of the mantissa for MXINT8 numbers can represent either 1. or 0., but for bfloat16 the 7th bit is always 1. (In other words, bfloat16 and MXINT8 numbers have different dynamic range). For instance:


MXINT8: 01111111 | 00.111100 = $2 ^ {2 ^ 7 - 1 - 127} \times (2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 0.9375 = 0.9375$

Output bfloat16 number (without bias subtraction): 0 | 01111111 | 1111000 = $2 ^ {2 ^ 7 - 1 - 127} \times (1 + 2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 1.9375 = 1.9375$

As we can see, the fact the bfloat16 representation assumes a leading 1. being added to the fractional value leads to dequantization error. Hence by subtracting the bias:

output - bias = 0 | 01111111 | 1111000 - 0 | 01111111 | 0000000 =

(0 | 00000000 | 1111000 - 0 | 00000000 | 1000000) << 1 + 0 | 01111111 | 0000000 - 0 | 00000001 | 0000000 = 0 | 01111110 | 1110000 = 0.9375

The computed value now matches the bfloat16 value. In fact, for all MXINT8 mantissa that is of the form X0XXXXXX, the dequantization error will be present and the bias subtraction will need to be done.
