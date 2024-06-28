from time import time

from random import random
import numpy as np
import math

ITERATIONS = int(1e7)

start = time()
buf = 0
for i in range(ITERATIONS):
    buf = np.hypot(1,2)
end = time()
print(f"A: {round(end-start, 6)}s")

start = time()
buf = 0
for i in range(ITERATIONS):
    buf = math.hypot(1,2)
end = time()
print(f"B: {round(end-start, 6)}s")