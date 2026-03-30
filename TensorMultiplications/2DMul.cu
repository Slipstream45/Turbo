#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n){
    int i = blockDim.x *blockIdx.x + threadIdx.x;
    if(i<n){
        C[i] = A[i]+B[i];
    }
}
__global__ void vectorMul(const float *A, const float *B, float *C, int M, int N, int P){
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i<M*P){
        float sum=0;
        int row = i/P;
        int col = i%P;

        for(int k=0;k<N;k++){
            sum += A[row*N+k] * B[k*P+col];
        }
        C[i] = sum;
    }
}

int main(){
    float A[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};
    float B[3][3] = {{9,8,7}, {6,5,4}, {3,2,1}};

    int rows=3,cols=3;
    int n = rows*cols;

    float *h_A = (float*)malloc(n*sizeof(float));
    float *h_B = (float*)malloc(n*sizeof(float));
    
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            h_A[i*cols+j] = A[i][j];
            h_B[i*cols+j] = B[i][j];
        }
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n*sizeof(float));
    cudaMalloc(&d_B, n*sizeof(float));
    cudaMalloc(&d_C, n*sizeof(float));

    cudaMemcpy(d_A, h_A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n*sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;

    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, 3,3,3);

    float h_C[9];
    cudaMemcpy(h_C, d_C, n*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            float value = h_C[i*cols+j];
            std::cout<<value<<" ";
        }
        std::cout<<std::endl;
    }
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
