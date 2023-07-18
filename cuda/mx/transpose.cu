#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

__global__ void transposeNative(float *input, float *output, int m, int n) {
    int colID_input = threadIdx.x + blockDim.x * blockIdx.x;
    int rowID_input = threadIdx.y + blockDim.y * blockIdx.y;
    if (rowID_input < m && colID_input < n) {
        int index_input = colID_input + rowID_input * n;
        int index_output = rowID_input + colID_input * m;
        output[index_output] = input[index_input];
    }
}


int main() {
    int m = 8192;
    int n = 4096;
    float *h_input, *h_output;
    cudaMallocHost(&h_input, m * n * sizeof(float));
    cudaMallocHost(&h_output, m * n * sizeof(float));
    random_init(h_input, m * n);
    float *d_input, *d_output;
    cudaMalloc(&d_input,  m * n * sizeof(float));
    cudaMalloc(&d_output,  m * n * sizeof(float));

    cudaMemcpy(d_input, h_input,  m * n * sizeof(float), cudaMemcpyDefault);
    
    dim3 block(8, 32);
    dim3 grid((n + block.x -1)/block.x, (m + block.y-1)/block.y);
    transposeNative<<<grid, block>>>(d_input, d_output, m , n);

    cudaMemcpy(h_output, d_output,  m * n * sizeof(float), cudaMemcpyDefault);

    cudaDeviceSynchronize();
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf(cudaGetErrorString(error));
        return -1;
    } else {
        printf("success \n");
    }
    return 0;
}