# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:41:50 2020

@author: nonli
"""

import numpy as np
import cupy as cp
import time

### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,1000))
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,1000))
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)