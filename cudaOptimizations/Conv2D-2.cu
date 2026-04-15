#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

#define TS 4
#define Radius 1
#define S (TS+2*Radius)

__global__ void conv2d_b(const float *input, const float *filter, float *output, int M, int N, int F){
   //high speed storage
   __shared__ float store[S][S];

   /*define dimensions, this approach is different, we no longer have 1:1 thread to pixel
   mapping anymore, so its possible 1 thread might bring 2 or more pixels (Halo ones)
   for that, we have to defind checkpoints with the blocks. That tells it to specifcially 
   release threads the way we want it, and the way we want it is to use Tile dims. 
   basically change 2D tile to 1D tile.
   */
   int checkpoint_y = blockIdx.y * TS;
   int checkpoint_x = blockIdx.x * TS;
   int ty = threadIdx.y;
   int tx = threadIdx.x;
   int thread_id = ty * TS + tx;

   /*now that we have a boundary defined, we can start pulling in pixels from input to the
   superfast cache __shared__ let's say 16 threads are there, so 16 threads will pull 16 
   pixels each iteration, so 0-15 pixels, then next it'll pull 16-31 pixels and so on..
   */
   for(int i=thread_id;i<S*S;i+=TS*TS){

        //need to point to the store's grid mapping
        int row = i/S;
        int col = i%S;

        /*this gives us a dynamic stride, now we have to fetch all the pixels, including
        the shadow ones
        */
        int global_row = checkpoint_y - Radius + row;
        int global_col = checkpoint_x - Radius + col;

        //job done, now copy
        if(global_row>=0 && global_row<M && global_col>=0 && global_row<N){
            store[row][col] = input[global_row * N + global_col];
        }
        else{
            store[row][col] = 0.0f;
        }
   }
   __syncthreads();

    float sum = 0.0f;
    for(int j =-1;j<2;j++){
        for(int i=-1;i<2;i++){
            float pixel = store[ty+Radius+j][tx+Radius+i];
            float weight = filter[(j+1)*F+(i+1)];
            sum += pixel*weight;
        }
    }

        //write back
    int out_row = checkpoint_y + ty;
    int out_col = checkpoint_x + tx;

    if(out_row<M && out_col<N){
        output[out_row *N + out_col] = sum;
    }
    __syncthreads();
}
