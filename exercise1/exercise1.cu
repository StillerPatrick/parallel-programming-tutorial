#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#define N 1000000

#define HANDLE_ERROR( err )(handleCudaError( err, __FILE__, __LINE__ ) )

int handleCudaError(cudaError_t cut,const char* file, int line)
{
	if(cut != cudaSuccess)
		{
		printf("%s %s %d \n",cudaGetErrorString(cut),file,line);
		return -1 ;
		}
	return 0;  
}


int main(int argc, char* argv[])
{
	// ============= INIT =====================
	int *a_host= NULL;
	int *b_host=NULL;
	int *a_device_ptr = NULL ;
	int *b_device_ptr = NULL ;

	 	
	

	a_host=(int*)malloc(sizeof(int)*N);
	b_host=(int*)malloc(sizeof(int)*N);

	assert(a_host);
	for(unsigned int i = 0; i < N; ++i)
	{
		a_host[i] = i ;
		b_host[i] = 0;
	}


	HANDLE_ERROR(cudaMalloc(&a_device_ptr, sizeof(int)*N)); // malloc of a_device
	HANDLE_ERROR(cudaMalloc(&b_device_ptr, sizeof(int)*N)); // malloc of b_device
	
	//Transfer a_host to a_device
	HANDLE_ERROR(cudaMemcpy(a_device_ptr, a_host, sizeof(int)*N, cudaMemcpyHostToDevice));

	//Transfer a_device to b_device

	HANDLE_ERROR(cudaMemcpy(b_device_ptr, a_device_ptr, sizeof(int)*N, cudaMemcpyDeviceToDevice));

	//Transfer b_device to b_host

	HANDLE_ERROR(cudaMemcpy(b_host, b_device_ptr, sizeof(int)*N, cudaMemcpyDeviceToHost));
	
	//transfercheck

	for(unsigned int i = 0; i < N; ++i)
	{
		//correct_transfer = correct_transfer & (a_host[i] == b_host[i]); 	
	if(a_host[i] != b_host[i]) {
		printf("Incorrect result at %d with %d and %d\n", i, a_host[i], b_host[i]);
		return -1 ;
		}
	}
	
	printf("Correct transfer \n");

	HANDLE_ERROR(cudaFree(a_device_ptr));
	HANDLE_ERROR(cudaFree(b_device_ptr));
	free(a_host);
	free(b_host);
	
	a_host= NULL;
	b_host= NULL; 

	return 0 ; 

}

