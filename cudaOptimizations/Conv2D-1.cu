__global__ void conv2d_a(const float *input, const float *filter, float *output, int M, int N, int F){
    //make space for the store:
    __shared__ float store[S][S];
    
    //get the dimensions
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    //using the tile, map the halos too
    int row = blockIdx.y * TS + ty - Radius;
    int col = blockIdx.x * TS + tx - Radius;

    /**disadvantage: we are using 36 threads for storing the pixels, but only 16 for
    actual computation**/

    //copy the pixels to the high speed storage
    if(row>=0 && row<M && col>=0 && col<N){
        store[ty][tx] = input[row*N+col];
    } else{
        store[ty][tx] = 0.0f;
    }
    __syncthreads();

    //compute
    if(ty>=Radius && ty<TS+Radius && tx>=Radius && tx<TS+Radius){

        float sum = 0.0f;
        for(int j =-1;j<2;j++){
            for(int i=-1;i<2;i++){
                float pixel = store[ty+j][tx+i];
                float weight = filter[(j+1)*F+(i+1)];
                sum += pixel*weight;
            }
        }

        //write back
        int out_row = blockIdx.y * TS + ty - Radius;
        int out_col = blockIdx.x * TS + tx - Radius;
        output[out_row *N + out_col] = sum;
    }
    __syncthreads();
}
