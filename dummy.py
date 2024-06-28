from time import time

import numpy as np
import math

ITERATIONS = int(1e7)

start = time()
buf = 0
for i in range(ITERATIONS):
    buf = i ** (1/3)
end = time()
print(f"A: {round(end-start, 6)}s")

start = time()
buf = 0
for i in range(ITERATIONS):
    buf = math.pow(i, 1/3)
end = time()
print(f"B: {round(end-start, 6)}s")

start = time()
buf = 0
for i in range(ITERATIONS):
    buf = math.cbrt(i)
end = time()
print(f"C: {round(end-start, 6)}s")