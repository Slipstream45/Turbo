#include <iostream>
#include <cuda_runtime.h>
__global__ void vecMul3D(const float *A, const float *B, float *C, int E, int M, int N, int P){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<E*M*P){
        float sum = 0.0f;

        //find outer of result:
        int z = i/(M*P);
        int rows = (i/P)%M;
        int cols = i%P;

        //find offset to specific 2D matrix:
        int A_outer = z*(M*N);
        int B_outer = z*(N*P);

        for(int k=0;k<N;k++){
            sum += A[A_outer+ (N*rows+k)] * B[B_outer+ (k*P+cols)];
        }
        C[i] = sum;
    }
}
__global__ void vecMul4D(const float *A, const float *B, float *C, int E, int F, int M, int N, int P){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<E*F*M*P){
        float sum = 0.0f;

        //find outer of result:
        int q = i/(F*M*P);
        int z = (i/(M*P))%F;
        int rows = (i/P)%M;
        int cols = i%P;

        //find offset to specific 2D matrix:
        int A_outer = q*(F*M*N) + z*(M*N);
        int B_outer = q*(F*N*P) + z*(N*P);

        for(int k=0;k<N;k++){
            sum += A[A_outer+ (N*rows+k)] * B[B_outer+ (k*P+cols)];
        }
        C[i] = sum;
    }
}
__global__ void vecMul5D(const float *A, const float *B, float *C, int E, int F, int G, int M, int N, int P){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<E*F*G*M*P){
        float sum = 0.0f;

        //find outer of result:
        int p = i/(F*G*M*P);
        int q = (i/(G*M*P))%F;
        int z = (i/(M*P))%G;
        int rows = (i/P)%M;
        int cols = i%P;

        //find offset to specific 2D matrix:
        int A_outer = p*(F*G*M*N) + q*(G*M*N) + z*(M*N);
        int B_outer = p*(F*G*N*P) + q*(G*N*P) + z*(N*P);

        for(int k=0;k<N;k++){
            sum += A[A_outer+ (N*rows+k)] * B[B_outer+ (k*P+cols)];
        }
        C[i] = sum;
    }
}

void handle3D(int E, int M, int N, int P){
    int n_A = E*M*N;
    int n_B = E*N*P;
    int n_C = E*M*P;

    size_t size_A = n_A*sizeof(float);
    size_t size_B = n_B*sizeof(float);
    size_t size_C = n_C*sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for(int i=0;i<E*M*N;i++){
        h_A[i] = 2.0f;
    }
    for(int i=0;i<E*N*P;i++){
        h_B[i] = 3.0f;
    }

    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    int number_of_threads = n_C;
    int threads_per_block = 256;
    int number_of_grids = (number_of_threads+threads_per_block-1)/threads_per_block;

    printf("Starting 3D Mat_Mul on GPU: \n");
    vecMul3D<<<number_of_grids, threads_per_block>>>(d_A, d_B, d_C, E, M, N, P);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    for(int i=0;i<5;i++){
        std::cout<<h_C[i]<<" ";
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}
void handle4D(int E, int F, int M, int N, int P){
    int n_A = E*F*M*N;
    int n_B = E*F*N*P;
    int n_C = E*F*M*P;

    size_t size_A = n_A*sizeof(float);
    size_t size_B = n_B*sizeof(float);
    size_t size_C = n_C*sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for(int i=0;i<E*F*M*N;i++){
        h_A[i] = 3.0f;
    }
    for(int i=0;i<E*F*N*P;i++){
        h_B[i] = 3.0f;
    }

    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    int number_of_threads = n_C;
    int threads_per_block = 256;
    int number_of_grids = (number_of_threads+threads_per_block-1)/threads_per_block;

    printf("\nStarting 4D Mat_Mul on GPU: \n");
    vecMul4D<<<number_of_grids, threads_per_block>>>(d_A, d_B, d_C, E, F, M, N, P);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    for(int i=0;i<5;i++){
        std::cout<<h_C[i]<<" ";
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}
void handle5D(int E, int F, int G, int M, int N, int P){
    int n_A = E*F*G*M*N;
    int n_B = E*F*G*N*P;
    int n_C = E*F*G*M*P;

    size_t size_A = n_A*sizeof(float);
    size_t size_B = n_B*sizeof(float);
    size_t size_C = n_C*sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    for(int i=0;i<E*F*G*M*N;i++){
        h_A[i] = 7.0f;
    }
    for(int i=0;i<E*F*G*N*P;i++){
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    int number_of_threads = n_C;
    int threads_per_block = 256;
    int number_of_grids = (number_of_threads+threads_per_block-1)/threads_per_block;

    printf("\nStarting 5D Mat_Mul on GPU: \n");
    vecMul5D<<<number_of_grids, threads_per_block>>>(d_A, d_B, d_C, E, F, G, M, N, P);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    for(int i=0;i<5;i++){
        std::cout<<h_C[i]<<" ";
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}

int main(){
    handle3D(3, 2, 5, 4); 
    handle4D(2, 3, 2, 10, 4); 
    handle5D(4, 2, 3, 2, 100, 4);
}
