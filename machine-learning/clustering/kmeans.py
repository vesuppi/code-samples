import numpy as np
from sklearn.cluster import KMeans
import torch
import triton
import triton.language as tl
from timeit import timeit
import cupy as cp
import torch.utils.benchmark as torchbench

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
        num_threads=torch.get_num_threads()
    )
    return t0.timeit(20).mean * 1000

np.random.seed(0)

M = 1024*256
N = 64
C = 16
X = np.random.randn(M, N).astype(np.float32)

init_centers = np.random.randn(C, N).astype(np.float32)
kmeans = KMeans(n_clusters=C, random_state=0, n_init=1, init=init_centers, max_iter=1)
kmeans.fit(X)
print(torch.get_num_threads())
print(kmeans.cluster_centers_, kmeans.n_iter_)
print(timeit(lambda: kmeans.fit(X), number=3)/3)
print('row-wise sum cpu time:', bench(lambda: np.sum(X, axis=1)))
print(bench(lambda: kmeans.fit(X)))


X = cp.array(X)
init_centers = cp.array(init_centers)
print(X.dtype)

src = open('kmeans.cu').read()
kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))

def kmeans():
    # Each block (`nthreads` threads) works on a row
    nthreads = N
    kernel(
        (M,), 
        (N,),
        (M, N, C, X, init_centers)
    )


kmeans()
ms, _, _ = triton.testing.do_bench(lambda: kmeans())
print(ms)

print('row-wise sum gpu time:', bench(lambda: cp.sum(X, axis=1)))