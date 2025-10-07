#!/usr/bin/env python3
import numpy as np
# add import and helper functions here
if __name__ == "__main__":
    print("hello world")
    np.random.seed(42)
    A = np.random.normal(size=(4, 4))
    B = np.random.normal(size=(4, 2))
    print(A @ B)
    np.random.seed(42)
    x = np.random.normal(size=(4, 10))
    D = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    print(D)
