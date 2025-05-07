#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

const int N = 50;
const int RANGE = 5;

// Kernel 1: Zero out the bucket
__global__ void init_bucket(int* bucket) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < RANGE) {
        bucket[i] = 0;
    }
}

// Kernel 2: Count frequency of each value
__global__ void count_instances(int* key, int* bucket, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        atomicAdd(&bucket[key[i]], 1);
    }
}

// Kernel 3: Write sorted values based on bucket counts and prefix offsets
__global__ void fill_sorted(int* key, int* bucket, int* prefix) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < RANGE) {
        for (int j = 0; j < bucket[i]; j++) {
            key[prefix[i] + j] = i;
        }
    }
}

int main() {
    int* key;
    int* bucket;
    int* prefix;

    // Allocate unified memory
    cudaMallocManaged(&key, N * sizeof(int));
    cudaMallocManaged(&bucket, RANGE * sizeof(int));
    cudaMallocManaged(&prefix, RANGE * sizeof(int));  // for offsets

    // Fill key with random values
    printf("Original: ");
    for (int i = 0; i < N; i++) {
        key[i] = rand() % RANGE;
        printf("%d ", key[i]);
    }
    printf("\n");

    // Kernel 1: Init buckets
    init_bucket<<<1, RANGE>>>(bucket);
    cudaDeviceSynchronize();

    // Kernel 2: Count frequencies
    count_instances<<<(N + 31) / 32, 32>>>(key, bucket, N);
    cudaDeviceSynchronize();

    // Prefix sum on CPU
    prefix[0] = 0;
    for (int i = 1; i < RANGE; i++) {
        prefix[i] = prefix[i - 1] + bucket[i - 1];
    }

    // Kernel 3: Fill sorted keys
    fill_sorted<<<1, RANGE>>>(key, bucket, prefix);
    cudaDeviceSynchronize();

    // Output sorted result
    printf("Sorted:   ");
    for (int i = 0; i < N; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    cudaFree(key);
    cudaFree(bucket);
    cudaFree(prefix);
    return 0;
}
