import cupy as cp

class KMeans():
    def __init__(self, n_clusters, init, max_iter):
        self.n_clusters = n_clusters
        self.init = cp.array(init)
        self.max_iter = max_iter

        self._update_labels = cp.RawKernel(open('update_labels.cu').read(), 'kernel', backend='nvcc', options=('-O3',))
        self._update_centers = cp.RawKernel(open('update_centers.cu').read(), 'kernel', backend='nvcc', options=('-O3',))
        self._aggregate_centers = cp.RawKernel(open('aggregate.cu').read(), 'kernel', backend='nvcc', options=('-O3',))

        
    def update_labels(self):
        M, N = self.X.shape
        nblocks = M
        nthreads = N
        self._update_labels(
            (nblocks,), 
            (nthreads,),
            (M, N, self.n_clusters, self.X, self.centers, self.labels)
        )

    def update_centers(self):
        M, N = self.X.shape
        nthreads = N
        self._update_centers(
            (self.nblocks_uc,), 
            (nthreads,),
            (M, N, self.n_clusters, M//self.nblocks_uc, self.X, self.block_centers, \
                self.block_center_counts, self.labels)
        )

    def aggregate_centers(self):
        M, N = self.X.shape
        nblocks = self.n_clusters
        nthreads = N
        self._aggregate_centers(
            (nblocks,), 
            (nthreads,),
            (M, N, self.n_clusters, self.nblocks_uc, self.X, \
                self.block_centers, self.block_center_counts, self.centers)
        )

    def setup(self, X):
        M, N = X.shape
        self.X = cp.array(X)
        self.centers = self.init
        self.labels = cp.empty([M], dtype=cp.int32)
        self.nblocks_uc = 256
        self.block_centers = cp.empty([self.nblocks_uc, self.n_clusters, N], dtype=X.dtype)
        self.block_center_counts = cp.empty([self.nblocks_uc, self.n_clusters], dtype=cp.int32)

    def fit(self, X):
        self.setup(X)
        for i in range(self.max_iter):
            self.update_labels()
            self.update_centers()
            self.aggregate_centers()
        self.update_labels()

        
        
        
