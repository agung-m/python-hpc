#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Mulya Agung

import cupy as cp


num_snps = 1000000
num_samples = 1000


start_gpu = cp.cuda.Event()
end_gpu = cp.cuda.Event()

start_gpu.record()
x = cp.random.randint(0, 4, num_snps * num_samples).astype('f4')
end_gpu.record()
end_gpu.synchronize()
t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

print("Elapsed time: {} ms".format(t_gpu))

