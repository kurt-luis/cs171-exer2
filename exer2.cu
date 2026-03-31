#include <iostream>
using namespace std;

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

int main()
{
    queryDevice();
    
    return 0;
}