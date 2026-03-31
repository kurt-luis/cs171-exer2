#include <iostream>
#include <random>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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
			sum = sum + A[A_idx] * B[B_idx];
		}

		int C_idx = row * M + col;
		C[C_idx] = sum;
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

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	matmul_rec_glob<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N, M, K);

	cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void printMatrix(const char* name, float* mat, int rows, int cols)
{
    printf("Matrix %s (%d x %d):\n", name, rows, cols);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int idx = r * cols + c;
            printf("%6.1f ", mat[idx]); 
        }
        printf("\n");
    }
    printf("--------------------------------------------------\n");
}

int main()
{
    queryDevice();
    
	int N = 3;
	int M = 4;
	int K = 2;

	float *A_h = new float[N * K];
	float *B_h = new float[K * M];

	// If A is NxK and B is KxM then C is NxM
	float *C_h = new float[N * M];

	randomizeElements(A_h, N, K);
	randomizeElements(B_h, K, M);

	printMatrix("A", A_h, N, K);
    printMatrix("B", B_h, K, M);

	matMul(A_h, B_h, C_h, N, M, K);

	for (int i = 0; i < N * M; i++)
	{
		printf("Element %d of matrix C is %.1f\n", i, C_h[i]);
	}

	delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    return 0;
}