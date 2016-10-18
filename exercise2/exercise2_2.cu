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

__global__ void parallel_scalar_product(int n, int *a,int *b, int *erg)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate current Thread
	if(i < n)
	{	
		int product = a[i] * b[i];
		atomicAdd(erg,product); // simple add 
	}


}

int main(int argc, char* argv[])
{
	// ============= INIT =====================
	int *a_host= NULL;
	int *b_host=NULL;
	int erg_host=0;
	int *a_device_ptr = NULL ;
	int *b_device_ptr = NULL ;
	int *erg_device_ptr = NULL ; 
	 	
	

	a_host=(int*)malloc(sizeof(int)*N);
	b_host=(int*)malloc(sizeof(int)*N);


	
	for(unsigned int i = 0; i < N; ++i)
	{
		a_host[i] = 1 ;
		b_host[i] = 1;
	}


	//============TRANSFER======================
	HANDLE_ERROR(cudaMalloc(&a_device_ptr, sizeof(int)*N)); // malloc of a_device
	HANDLE_ERROR(cudaMalloc(&b_device_ptr, sizeof(int)*N)); // malloc of b_device
	HANDLE_ERROR(cudaMalloc(&erg_device_ptr, sizeof(int))); // malloc of erg_device 
	
	//Transfer a_host to a_device
	HANDLE_ERROR(cudaMemcpy(a_device_ptr, a_host, sizeof(int)*N, cudaMemcpyHostToDevice));
		
	//Transfer b_host to b_device
	HANDLE_ERROR(cudaMemcpy(b_device_ptr, b_host, sizeof(int)*N, cudaMemcpyHostToDevice));
	
	HANDLE_ERROR(cudaMemcpy(erg_device_ptr, &erg_host, sizeof(int), cudaMemcpyHostToDevice));

	
	//=============Calculation ==================
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	cudaEventRecord(start);
	parallel_scalar_product<<<ceil((float)N/(float)256),256>>>(N,a_device_ptr,b_device_ptr,erg_device_ptr);
	cudaEventRecord(stop);



	//===========CHECK============================
 
	
	HANDLE_ERROR(cudaMemcpy(&erg_host,erg_device_ptr, sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time %f milliseconds \n", milliseconds) ; 

/*	int erg = 0
	for(unsigned int i = 0; i < N; ++i)
	{
		//correct_transfer = correct_transfer & (a_host[i] == b_host[i]); 	
		erg += a[i] *b[i] ;

	} */

	if(erg_host == N)
	{
	printf("Correct Calculation \n");
	
	}
	else
	{
	printf(" Non Correct Calculation %d %d \n", erg_host , N);
	}
 	//============CLEAN==============================

	HANDLE_ERROR(cudaFree(a_device_ptr));
	HANDLE_ERROR(cudaFree(b_device_ptr));
	HANDLE_ERROR(cudaFree(erg_device_ptr));
	free(a_host);
	free(b_host);
	
	a_host= NULL;
	b_host= NULL; 

	return 0 ; 

}

