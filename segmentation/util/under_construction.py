import numpy as np

def return_array():
    arr = np.random.randn(10)
    print(arr.shape, arr)
    return arr

out = return_array()
print(f'array outside: {out.shape} with {out}')