#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

using namespace std;

template <typename T>
cudaError_t allocateDevice(T** devPtr, size_t size)
{
	return cudaMalloc((void**)devPtr, size);
}

void queryDevice()
{
    int deviceId;
	
	cudaGetDevice(&deviceId);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceId);
	
	cout << "========================DEVICE PROPERTIES=========================\n";
	printf("Device Name: %s\n", props.name);
	printf("Compute Capability: %d.%d\n", props.major, props.minor);
	printf("Total VRAM: %zu GB\n", props.totalGlobalMem / (1024 * 1024 * 1024));
	printf("Warp size: %d\n", props.warpSize);
	printf("SM Count: %d\n", props.multiProcessorCount);
	printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
	cout << "==================================================================\n";
}

void randomizeElements(float *mat, int n, int m)
{
	random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<int> dist(0, 10); 

    for (int i = 0; i < n * m; i++)
    {
        mat[i] = static_cast<float>(dist(gen)); 
    }
}

__global__ void matmul_rec_glob(float *A, float *B, float *C, int N, int M, int K)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < M)
	{
		float sum = 0;

		for (int i = 0; i < K; i++)
		{
			int A_idx = row * K + i;
			int B_idx = col + i * M;
			sum += A[A_idx] * B[B_idx];
		}

		int C_idx = row * M + col;
		C[C_idx] = sum;
	}
}

__global__ void matmul_rec_shar(float *A, float *B, float *C, int N, int M, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];
	
    float sum = 0.0f;

    for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; i++)
    {
		int A_col = i * TILE_SIZE + threadIdx.x;

		if (row < N && A_col < K)
		{
			A_shared[threadIdx.y][threadIdx.x] = A[row * K + A_col];
		}

		else
		{
			A_shared[threadIdx.y][threadIdx.x] = 0.0f;
		}


		int B_row = i * TILE_SIZE + threadIdx.y;

		if (B_row < K && col < M)
		{
			B_shared[threadIdx.y][threadIdx.x] = B[B_row * M + col];
		}

		else
		{
			B_shared[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();
        
        for (int j = 0; j < TILE_SIZE; j++)
		{
			sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
		}

		__syncthreads();
    }

    if (row < N && col < M)
    {
        C[row * M + col] = sum;
    }
}

void matMul(float *A_h, float *B_h, float *C_h, int N, int M, int K)
{
	float *A_d, *B_d, *C_d;

	allocateDevice(&A_d, N * K * sizeof(float));
    allocateDevice(&B_d, K * M * sizeof(float));
    allocateDevice(&C_d, N * M * sizeof(float));

	cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float totalTimeGlobal = 0.0f, totalTimeShared = 0.0f;
	int numRuns = 10;

	for (int i = 0; i < numRuns; i++)
	{
		cudaEventRecord(start);

		matmul_rec_glob<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N, M, K);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		totalTimeGlobal += milliseconds;


		cudaEventRecord(start);

		matmul_rec_shar<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N, M, K);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		totalTimeShared += milliseconds;		
	}

	float avgTimeGlobal = totalTimeGlobal / numRuns;
	float avgTimeShared = totalTimeShared / numRuns;

	printf("Matrix Multiplication of %d x %d and %d x %d.\nGlobal Average: %f ms, Shared Average: %f ms\n\n", N, K, K, M, avgTimeGlobal, avgTimeShared);

	cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    queryDevice();

	vector<vector<int>> test_dimensions = {
		{256, 256, 256},
		{256, 256, 512},
		{512, 512, 512},
		{512, 512, 1024},
		{1024, 1024, 1024},
		{1024, 1024, 2048},
		{2048, 2048, 2048},
	};

	int N, M, K;

	for (int test = 0; test < 7; test++)
	{
		printf("Test %d:\n", test + 1);
		
		N = test_dimensions[test][0];
		M = test_dimensions[test][1];
		K = test_dimensions[test][2];

		float *A_h = new float[N * K];
		float *B_h = new float[K * M];

		// If A is NxK and B is KxM then C is NxM
		float *C_h = new float[N * M];

		randomizeElements(A_h, N, K);
		randomizeElements(B_h, K, M);

		matMul(A_h, B_h, C_h, N, M, K);

		delete[] A_h;
		delete[] B_h;
		delete[] C_h;
	}
    

    return 0;
}