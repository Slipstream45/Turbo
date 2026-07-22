__global__ void tiled_coarsed_gemm(const float *A, const float *B, float *C, int M, int N, int K){
    __shared__ float Ads[TS][TS];
    __shared__ float Bds[TS][TS];
    int by = blockIdx.y; int bx = blockIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;
    int Row = by*TS+ty;
    int Col = bx*TS+tx;
    float pVal0 =0.0f;
    float pVal1 =0.0f;
    float pVal2 = 0.0f;
    float pVal3 = 0.0f;

    for(int ph=0;ph<(K+TS-1)/TS;ph++){
        for(int i=0;i<CF;i++){
            if(Row<M && (ph*TS+(tx+i*blockDim.x))<K){
                Ads[ty][tx+(i*blockDim.x)] = A[Row*K+(ph*TS+(tx+(i*blockDim.x)))];
            }else{
                Ads[ty][tx+(i*blockDim.x)] = 0.0f;
            }
        }
        for(int j=0;j<CF;j++){
            if((ty+j*blockDim.y)<K && Col<N){
                Bds[ty+(j*blockDim.y)][tx] = B[(ph*TS+(ty+(j*blockDim.y)))*N+Col];
            }else{
                Bds[ty+(j*blockDim.y)][tx] = 0.0f;
            }
        }
        __syncthreads();

        for(int k=0;k<TS;k++){

            float a1 = Ads[ty][k];
            float a2 = Ads[ty+blockDim.y][k];
            float b1 = Bds[k][tx]; 
            float b2 = Bds[k][tx+blockDim.x];
            pVal0 += a1 * b1;
            pVal1 += a2 * b1;
            pVal2 += a1 * b2;
            pVal3 += a2 * b2;
        }
        __syncthreads();
    }
    int r1 = Row; int r2 = Row+blockDim.y;
    int c1 = Col; int c2 = Col+blockDim.x;
    if(r1<M && c1<N){
        C[r1*N+c1] = pVal0;
    }
    if(r2<M && c1<N){
        C[r2*N+c1] = pVal1;
    }
    if(r1<M && c2<N){
        C[r1*N+c2] = pVal2;
    }
    if(r2<M && c2<N){
        C[r2*N+c2] = pVal3;
    }
}


for(int ph=0;ph<(K+TS-1)/TS;ph++){
    for(int i=0;i<CF;i++){
        Ads[ty][tx+(i*blockDim.x)] = A[Row*K+(ph*TS+(tx+(i*blockDim.x)))];
    }
    for(int j=0;j<CF;j++){
        Bds[ty+(j*blockDim.y)][tx] = B[(ph*TS+(ty+(j*blockDim.y)))*N+Col];
    }
}
