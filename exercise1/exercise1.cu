#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#define N 1000000


int main(int argc, char* argv[])
{
	// ============= INIT =====================
	int *a_host= NULL;
	int *b_host=NULL;
	int *a_device_ptr = NULL ;
	int *b_device_ptr = NULL ;
	int correct_transfer = 1 ;
	

	a_host=(int*)malloc(sizeof(int)*N);
	b_host=(int*)malloc(sizeof(int)*N);

	assert(a_host);
	for(unsigned int i = 0; i < N; ++i)
	{
		a_host[i] = i ;
	}
	a_host=(int*)malloc(sizeof(int)*N);
	b_host=(int*)malloc(sizeof(int)*N);

	cudaMalloc(&a_device_ptr, sizeof(int)*N); // malloc of a_device
	cudaMalloc(&b_device_ptr, sizeof(int)*N); // malloc of a_device

	//Transfer a_host to a_device
	cudaMemcpy(a_device_ptr, a_host, sizeof(int)*N, cudaMemcpyHostToDevice);

	//Transfer a_device to b_device

	cudaMemcpy(b_device_ptr, a_device_ptr, sizeof(int)*N, cudaMemcpyDeviceToDevice);

	//Transfer b_device to b_host

	cudaMemcpy(b_host, b_device_ptr, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	//transfercheck

	for(unsigned int i = 0; i < N; ++i)
	{
		correct_transfer = correct_transfer & (a_host[i] == b_host[i]); 	
	}

	if(correct_transfer)
	{
		printf("Correct Transfer");
	}


	cudaFree(a_device_ptr);
	cudaFree(b_device_ptr);
	free(a_host);
	free(b_host);
	
	a_host= NULL;
	b_host= NULL; 

}
