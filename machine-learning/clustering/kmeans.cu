/**
Reduction using warp reduction instructions. This approach uses less
shared memory than previous approaches.
 */
 __inline__ __device__ float warpReduce(float value) {
    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    // "value" now contains the sum across all threads
    //printf("Thread %d final value = %d\n", threadIdx.x, value);
    return value;
}

__inline__ __device__ float blockReduce(float sum) {
    sum = warpReduce(sum);
    int tid = threadIdx.x;
    __shared__ float psums[16];
    if (tid % 32 == 0) {
        psums[tid / 32] = sum;
    }
    __syncthreads();

    sum = 0;
    for (int i = 0; i < blockDim.x / 32; i++) {
        sum += psums[i];
    }
    return sum;
}

extern "C" __global__
void kernel(int M, int N, int C, int BLOCK_M, float* X, float* centers,
    float* block_centers, float* block_center_counts, int* assigns) {
    int om = blockIdx.x;
    int n = threadIdx.x;

    int bound = min(om*BLOCK_M+BLOCK_M, M);

    float* block_centers_start = block_centers + blockIdx.x * C * N;
    for (int c = 0; c < C; c++) {
        block_centers_start[c*N+n] = 0;
        if (n == 0) {
            block_center_counts[blockIdx.x*C + c] = 0;
        }
    }

    for (int m = om*BLOCK_M; m < bound; m++) {
        float x = X[m*N + n];
        float min_dist = 100000000;
        int min_dist_cluster = -1;
        for (int i = 0; i < C; i++) {
            float c = centers[i*N + n];
            float t = (x-c) * (x-c);
    
            float sum = blockReduce(t);
            if (n == 0) {
                float distance = sqrt(sum);
                if (distance < min_dist) {
                    min_dist = distance;
                    min_dist_cluster = i;
                }
            }
        }
        if (n == 0) {
            assigns[m] = min_dist_cluster;
        }

        int c = min_dist_cluster;
        block_centers_start[c*N+n] += x;
        if (n == 0) {
            block_center_counts[blockIdx.x*C + c] += 1;
        }
    }

}