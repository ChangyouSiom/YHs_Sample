#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A, const float *B, const float *C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i*k +p] * B[j+p*n];
            }
            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

__global__ void sgemm_kernel(const float *A, const float *B, float *C, int M, int N , int K) {
    int m = blockIdx.y * 128 + threadIdx.x / 2;
    int n = blockIdx.x * 256 + (threadIdx.x % 2) * 128;

    #pragma unroll
    for (int j=0; j<128; ++j) {
        int index = m*N + n + j;
        if (index < M * N){
            C[index] = 0;
        } else {
            printf("index error: %d \n", index);
        }

    }

    #pragma unroll
    for (int i=0; i<K; ++i) {
        #pragma unroll
        for (int j=0; j<128; ++j) {
            int index = m*N + n + j;
            if (index < M * N){
                C[index] += A[m*K + i] * B[i*N+n+j];
            } else {
                printf("index error: %d \n", index);
            }

        }
    }
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + 255) / 256, (m + 127) /128);//列数， 行数

    //warmup
    sgemm_kernel<<<grid, 256>>>(d_A, d_B, d_C, m, n, k);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        sgemm_kernel<<<grid, 256>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("Time: %f ms\n", ms/n_iter);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    // cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // bool chk = check(h_A, h_B, h_C, m, n, k);
    // printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return 0;
}