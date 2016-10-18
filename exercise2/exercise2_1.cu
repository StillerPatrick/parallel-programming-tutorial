#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#define N 1000000

#define HANDLE_ERROR( err )(handleCudaError( err, __FILE__, __LINE__ ) )

int handleCudaError(cudaError_t cut,const char* file, int line)
{
	if(cut != cudaSuccess)
		{
		printf("%s : File: %s  Line: %d \n",cudaGetErrorString(cut),file,line);
		return -1 ;
		}
	return 0;  
}


__global__ void parallel_add(int n, int *a ,int *b , int *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate current Thread
	if(i < n)
	{
		c[i] = a[i] + b[i]; // simple add 
	}
}

int main(int argc, char* argv[])
{
	// ============= INIT =====================
	int *a_host= NULL;
	int *b_host=NULL;
	int *c_host=NULL;
	int *a_device_ptr = NULL ;
	int *b_device_ptr = NULL ;
	int *c_device_ptr = NULL ; 
	 	
	

	a_host=(int*)malloc(sizeof(int)*N);
	b_host=(int*)malloc(sizeof(int)*N);
	c_host=(int*)malloc(sizeof(int)*N);

	
	for(unsigned int i = 0; i < N; ++i)
	{
		a_host[i] = i ;
		b_host[i] = i;
	}

	//============TRANSFER======================
	HANDLE_ERROR(cudaMalloc(&a_device_ptr, sizeof(int)*N)); // malloc of a_device
	HANDLE_ERROR(cudaMalloc(&b_device_ptr, sizeof(int)*N)); // malloc of b_device
	HANDLE_ERROR(cudaMalloc(&c_device_ptr, sizeof(int)*N)); // malloc of c_device 
	
	//Transfer a_host to a_device
	HANDLE_ERROR(cudaMemcpy(a_device_ptr, a_host, sizeof(int)*N, cudaMemcpyHostToDevice));
		
	//Transfer b_host to b_device
	HANDLE_ERROR(cudaMemcpy(b_device_ptr, b_host, sizeof(int)*N, cudaMemcpyHostToDevice));
	

	
	//=============Calculation ==================
	
	
	parallel_add<<<ceil((float)N/(float)256>>>(N,a_device_ptr,b_device_ptr,c_device_ptr);




	//===========CHECK============================
 
	
	HANDLE_ERROR(cudaMemcpy(c_host,c_device_ptr, sizeof(int)*N, cudaMemcpyDeviceToHost));

	for(unsigned int i = 0; i < N; ++i)
	{
		//correct_transfer = correct_transfer & (a_host[i] == b_host[i]); 	
		if(c_host[i] != (a_host[i]+ b_host[i])) {
			printf("Incorrect result at %d with %d and %d\n", i, a_host[i], b_host[i]);
			return -1 ;
		}
	}
	
	printf("Correct Calculation \n");

 	//============CLEAN==============================

	HANDLE_ERROR(cudaFree(a_device_ptr));
	HANDLE_ERROR(cudaFree(b_device_ptr));
	HANDLE_ERROR(cudaFree(c_device_ptr));
	free(a_host);
	free(b_host);
	
	a_host= NULL;
	b_host= NULL; 

	return 0 ; 

}

