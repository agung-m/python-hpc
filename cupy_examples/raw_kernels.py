#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Mulya Agung

from cupyx import jit
import cupy


@jit.rawkernel()
def elementwise_ols(x, y, b, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x

    for i in range(tid, size, ntid):
        b[i] = y[i] / x[i]
        #b[i] = squared_diff(x[i], y[i])
    #b = cupy.linalg.lstsq(x, y)


size = cupy.uint32(2 ** 22)
x = cupy.random.normal(size=(size,), dtype=cupy.float32)
y = cupy.random.normal(size=(size,), dtype=cupy.float32)
b = cupy.empty((size,), dtype=cupy.float32)

elementwise_ols((128, ), (1024, ), (x, y, b, size))

#assert (x == y).all()
print(b)
