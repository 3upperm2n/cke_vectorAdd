#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

const int N = 1 << 20;

#define FLTSIZE sizeof(float)

inline int BLK(int data, int blocksize)
{
	return (data + blocksize - 1) / blocksize;
}

__global__ void kernel_vectorAdd (const float* __restrict__ a_d, 
		const float* __restrict__ b_d,
		const int N,
		float *c_d)
{
	int tid = threadIdx.x + __mul24(blockIdx.x, blockDim.x);

	if(tid < N) {
		c_d[tid] = a_d[tid] + b_d[tid];	
	}
}

int main( int argc, char **argv)
{
	int num_streams = 8;

	if(argc == 2)
		num_streams = atoi(argv[1]);

	int num_threads = num_streams;

	printf("\nrunning %d cuda streams (and threads)\n", num_streams);

	// allocate streams
    cudaStream_t *streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));

	// init
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }
	
	//------------------------------------------------------------------------//
	// allocate data on the host
	//------------------------------------------------------------------------//
	size_t databytes = N  * FLTSIZE; 

	float *a_h = (float*) malloc ( N * num_streams * FLTSIZE);
	float *b_h = (float*) malloc ( N * num_streams * FLTSIZE);
	float *c_h = (float*) malloc ( N * num_streams * FLTSIZE);

	for(int i=0; i< N * num_streams; i++) {
		a_h[i] = 1.1f;	
		b_h[i] = 2.2f;	
	}

	//------------------------------------------------------------------------//
	// allocate data on the device 
	//------------------------------------------------------------------------//
	float *a_d;
	float *b_d;
	float *c_d;
	cudaMalloc((void**)&a_d, N * num_streams * FLTSIZE);
	cudaMalloc((void**)&b_d, N * num_streams * FLTSIZE);
	cudaMalloc((void**)&c_d, N * num_streams * FLTSIZE);

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        checkCudaErrors(cudaSetDevice(0));

		int id = omp_get_thread_num();
		size_t offset = id * N;

		// kernel configuration
		dim3 threads = dim3(256, 1, 1);
		dim3 blocks  = dim3(BLK(N, threads.x), 1, 1);

		// copy data to deivce
		cudaMemcpyAsync(a_d + offset, a_h + offset,  databytes, cudaMemcpyHostToDevice, streams[id]);
		cudaMemcpyAsync(b_d + offset, b_h + offset,  databytes, cudaMemcpyHostToDevice, streams[id]);

		// launch one worker kernel per stream
		kernel_vectorAdd <<< blocks, threads, 0, streams[id] >>> (&a_d[offset], 
				                                                 &b_d[offset], 
																 N, 
																 &c_d[offset]);

		// copy data back to host
		cudaMemcpyAsync(c_h + offset, c_d + offset,  databytes, cudaMemcpyDeviceToHost, streams[id]);
	}

	// check data
	bool success = 1;
	for(int i=0; i< N * num_streams; i++) {
		if (abs(c_h[i] - 3.3f) > 1e-6) {
			fprintf(stderr, "%d : %f  (error)!\n", i, c_h[i]);
			success = 0;
			break;
		}
	}

	if(success) {
		printf("\nSuccess!\nExit.\n");	
	}

	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

	free(a_h);
	free(b_h);
	free(c_h);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	cudaDeviceReset();

	return 0;
}
