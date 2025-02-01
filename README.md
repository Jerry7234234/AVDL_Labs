# AVDL Labs

A brief description of the project.

## Table of Contents
- [Lab4](#Lab4)

## Lab4
### Question: What is the purpose of the dont_need_abs variable and the bias variable? Note that unlike IEEE Floating-Point, MXINT has no implicit leading bit for the mantissa.
During the conversion from MXINT8 to bfloat16, only the last 6 bits of the mantissas are used in the fraction part of the output. The 7th bit is ignored and the extracted 6 bits are left shifted by 1 to increase the bit length to 7. This can cause dequantization errors as the 7th bit of the mantissa for MXINT8 numbers can represent either 1. or 0. For instance:


MXINT8 = 01111111 | 00.111100 = $2 ^ {2 ^ 7 - 1 - 127} \times (2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 1 \times 0.9375 = 0.9375$
