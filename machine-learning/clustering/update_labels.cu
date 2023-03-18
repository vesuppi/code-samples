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
void kernel(int M, int N, int C, float* X, float* centers, int* labels) {
    int m = blockIdx.x;
    int n = threadIdx.x;

    float x = X[m*N + n];
    float min_dis = 1000000;
    float cluster = -1;
    for (int i = 0; i < C; i++) {
        float c = centers[i*N + n];
        float pdis = (x-c) * (x-c);
        float dis = blockReduce(pdis);
        if (dis < min_dis) {
            min_dis = dis;
            cluster = i;
        }
    }

    if (n == 0) {
        labels[m] = cluster;
    }
}