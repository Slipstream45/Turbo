/* * WARNING: This is a pure-physics implementation of Tiled Matrix Multiplication.
 * It assumes ideal hardware alignment. Matrix dimensions (M, N, P) MUST be 
 * perfect multiples of the Tile Size (TS). 
 * Valid dimensions for TS=32: 32, 64, 96, 128, 1024, etc. 
 */
#include <iostream>
#include <cuda_runtime.h>

#define TS 32

__global__ void Tiled_mat_mul(const float *A, const float *B, float *C, int M, int N, int P){
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float pSum = 0.0f;
    int steps = (N)/TS;

    for(int i=0;i<steps;i++){
        //load
        As[ty][tx] = A[row*N + (i*TS+tx)];
        Bs[ty][tx] = B[(i*TS+ty)*P + col];

        __syncthreads();

        //compute
        for(int k=0;k<TS;k++){
            pSum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row*P+col] = pSum;

}

int main(){
    int M = 512;
    int N = 1024;
    int P = 256;
    size_t size_A = M*N*sizeof(float);
    size_t size_B = N*P*sizeof(float);
    size_t size_C = M*P*sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    for(int i = 0; i < M * N; i++) h_A[i] = 2.0f;
    for(int i = 0; i < N * P; i++) h_B[i] = 2.0f;

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TS, TS);
    dim3 blocksPerGrid((P + TS - 1) / TS, (M + TS - 1) / TS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 2. Start the timer exactly before the launch
    cudaEventRecord(start);

    Tiled_mat_mul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for the click

    // 5. Calculate the exact milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\n[PROFILER] GPU Kernel Execution Time: " << milliseconds << " ms\n" << std::endl;
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++){
            float value = h_C[i*P+j];
            //std::cout<<value<<" ";
        }
        //std::cout<<std::endl;
    }
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
