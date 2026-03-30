%%writefile kernels.cu

extern "C" {

  #define DIMS 30
  #define PI 3.1415926535f

  __global__ void rastriginKernel(const float *vector, float *energy, int threads){
      int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if(t_idx<threads){
          float e = 10.0f*DIMS;
          for(int i=0;i<DIMS;i++){
              int flat_idx = (t_idx*DIMS)+i;
              float x = vector[flat_idx];
              float termi = (x*x)-10.0f*cosf(2.0f*PI*x);
              e+=termi;
          }
          energy[t_idx] = e;
      }
  }

  __global__ void sphereKernel(const float *vector, float *energy, int threads){
      int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if(t_idx<threads){
          float e = 0.0f;
          for(int i=0;i<DIMS;i++){
              int flat_idx = (t_idx*DIMS)+i;
              float x = vector[flat_idx];

              e += (x*x);
          }
          energy[t_idx] = e;
      }
  }

    __global__ void ackleyKernel(const float *vector, float *energy, int threads){
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx<threads){
            float a =20.0f;
            float sq_sum=0.0f;
            float sum_cos=0.0f;
            float e = 0.0f;
            for(int i=0;i<DIMS;i++){
                int flat_idx = (t_idx*DIMS)+i;
                float x = vector[flat_idx];
                sq_sum += (x*x);
                sum_cos += cosf(2.0f*PI*x);
            }
            float termi1 = -a*expf(-0.2f*sqrtf(sq_sum/(float)DIMS));
            float termi2 = -expf(sum_cos/(float)DIMS);
            e = termi1 + termi2 + a + expf(1.0f);
            energy[t_idx] = e;
        }
    }
    __global__ void rosenbrockKernel(const float *vector, float *energy, int threads){
        int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(t_idx<threads){
            float a = 1.0f;
            float b = 100.0f;
            float e = 0.0f;
            for(int i=0;i<DIMS-1;i++){
                int flat_idx = (t_idx*DIMS)+i;
                float x_i = vector[flat_idx];
                float x_next = vector[flat_idx+1];

                float diff1 = x_next  -(x_i*x_i);
                float diff2 = a-x_i;
                float termi1 = b*(diff1*diff2);
                float termi2 = diff2*diff2;
                e += termi1+termi2;
            }
            energy[t_idx] = e;
        }
    }
    /*
    int main(){
        int num_threads = 100000;
        size_t vector_size = num_threads*DIMS*sizeof(float);
        size_t energy_size = num_threads*sizeof(float);

        float *h_vec = (float*)malloc(vector_size);
        float *h_eng = (float*)malloc(energy_size);

        for(int i =0;i<num_threads*DIMS;i++){
            h_vec[i] = 0.05f;
        }
        float *d_vec;
        cudaMalloc((void**)&d_vec, vector_size);
        float *d_eng;
        cudaMalloc((void**)&d_eng, energy_size);

        cudaMemcpy(d_vec, h_vec, vector_size, cudaMemcpyHostToDevice);
        int threads_per_block = 256;
        int block_per_grid = (num_threads + threads_per_block-1)/threads_per_block;

        rastriginKernel<<<block_per_grid, threads_per_block>>>(d_vec, d_eng, num_threads);
        cudaMemcpy(h_eng, d_eng, energy_size, cudaMemcpyDeviceToHost);
        std::cout<<"Result Verified. Thread 0 energy: "<<h_eng[0]<<std::endl;
        cudaFree(d_vec);
        cudaFree(d_eng);
        free(h_vec);
        free(h_eng);
        return 0;
    }
    */
}
