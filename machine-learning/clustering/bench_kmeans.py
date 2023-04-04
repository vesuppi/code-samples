import numpy as np
from sklearn.cluster import KMeans
from kmeans import KMeans as GPUKmeans
import torch
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

M = 60000
N = 512
C = 128
X = np.random.randn(M, N).astype(np.float32)

init_centers = np.random.randn(C, N).astype(np.float32)
max_iters = 20
kmeans = KMeans(n_clusters=C, random_state=0, n_init=1, init=init_centers, max_iter=max_iters)
kmeans.fit(X)

print('kmeans centers:', kmeans.cluster_centers_[0])
print('kmeans labels:', kmeans.labels_)
print('kmeans iters:', kmeans.n_iter_)
#print(kmeans.cluster_centers_, kmeans.n_iter_)
#print(timeit(lambda: kmeans.fit(X), number=3)/3)
#print('row-wise sum cpu time:', bench(lambda: np.sum(X, axis=1)))
print('kmeans sklearn:', bench(lambda: kmeans.fit(X))/kmeans.n_iter_)
#print('kmeans centers:', kmeans.cluster_centers_)

gkmeans = GPUKmeans(n_clusters=C, init=init_centers, max_iter=max_iters)
gkmeans.fit(X)
print(cp.sum(cp.array(kmeans.labels_)- gkmeans.labels))
print(gkmeans.centers[0])
print(gkmeans.labels)

gkmeans.setup(X)
print(bench(lambda: gkmeans.update_labels()))
print(bench(lambda: gkmeans.update_centers()))
print(bench(lambda: gkmeans.aggregate_centers()))
print(bench(lambda: gkmeans.fit(X))/max_iters)
#gkmeans.setup(X)

#print(bench(lambda: gkmeans.update_labels()))

# _update_labels = cp.RawKernel(open('update_labels.cu').read(), 'kernel', backend='nvcc', options=('-O3',))
# _update_centers = cp.RawKernel(open('update_centers.cu').read(), 'kernel', backend='nvcc', options=('-O3',))
# _aggregate_centers = cp.RawKernel(open('aggregate.cu').read(), 'kernel', backend='nvcc', options=('-O3',))


# def update_labels(centers, labels):
#     nblocks = M
#     nthreads = N
#     _update_labels(
#         (nblocks,), 
#         (nthreads,),
#         (M, N, C, X, centers, labels)
#     )


# nblocks_uc = 128
# centers = cp.empty([nblocks_uc, C, N], dtype=X.dtype)
# center_counts = cp.empty([nblocks_uc, C], dtype=cp.int32)        
# def update_centers(labels):
#     nthreads = N
#     _update_centers(
#         (nblocks_uc,), 
#         (nthreads,),
#         (M, N, C, M//nblocks_uc, X, centers, center_counts, labels)
#     )


# def aggregate_centers(final_centers):
#     nblocks = C
#     nthreads = N
#     _aggregate_centers(
#         (nblocks,), 
#         (nthreads,),
#         (M, N, C, nblocks_uc, X, centers, center_counts, final_centers)
#     )

# def kmeans_gpu(iters=max_iters):
#     gpu_init_centers = cp.array(init_centers)
#     labels = cp.empty([M], dtype=cp.int32)
#     for i in range(iters):
#         update_labels(centers=gpu_init_centers, labels=labels)
#         update_centers(labels)
#         aggregate_centers(gpu_init_centers)
#     return labels

# labels = kmeans_gpu(max_iters)
# print(bench(lambda: kmeans_gpu(max_iters))/max_iters)
# print(labels)
# print('labels allclose:', cp.allclose(labels, kmeans.labels_))

# update_labels()
# print('update labels:', bench(lambda: update_labels()))
# print(labels)

# update_centers()
# print('update centers:', bench(lambda: update_centers()))
# print(center_counts[0])

# aggregate_centers()
# print('aggregate centers:', bench(lambda: aggregate_centers()))

# update_labels()
# print('labels allclose:', cp.allclose(labels, kmeans.labels_))



#print('our centers:', init_centers)
# assigns = cp.empty([M], dtype=cp.int32)
# BLOCK_M = 128*4
# nblocks = M // BLOCK_M
# nthreads = N
# print(nblocks)
# block_centers = cp.empty([nblocks, C, N], dtype=X.dtype)
# block_center_counts = cp.empty([nblocks, C], dtype=X.dtype)
# def kmeans_gpu():
#     # Each block (`nthreads` threads) works on a row
    
#     #block_centers = cp.empty([nblocks, C, N])
#     kernel(
#         (nblocks,), 
#         (nthreads,),
#         (M, N, C, BLOCK_M, X, init_centers, block_centers, block_center_counts, assigns)
#     )


# kmeans_gpu()
# print(block_center_counts)
# # print(assigns)
# # for i in range(M):
# #     if kmeans.labels_[i] != assigns[i]:
# #         print(i, kmeans.labels_[i], assigns[i])
# # print(cp.allclose(assigns, cp.array(kmeans.labels_)))
# print('kmeans gpu:', bench(lambda: kmeans_gpu()))

# #print('row-wise sum gpu time:', bench(lambda: cp.sum(X, axis=1)))