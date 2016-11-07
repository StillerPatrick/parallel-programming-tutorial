#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>


#define HANDLE_ERROR( err )(handleCudaError( err, __FILE__, __LINE__ ) )





__global__ void diadic_Product (int n, int *a,int *b, int *erg)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate current Thread in x
	int j = blockIdx.y * blockDim.y + threadIdx.y; // Calculate current Tread in y
	
	if(i < n && j < n)
	{	
		
		erg[i*n+j] = a[i]*b[j]; 	
	
	}
}

__global__ void matrixProduct(int n, int *a, int *b, int *c)
{
		int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate current Thread in x
		int j = blockIdx.y * blockDim.y + threadIdx.y; // Calculate current Tread in y
		
		int scalarProduct = 0; 
		for(int k = 0; k < n; ++k)
		{
			scalarProduct += a[i*n+k]* b[k*n+j];
		}
		c[i*n+j] = scalarProduct ;
	
	
}

int handleCudaError(cudaError_t cut,const char* file, int line)
{
	if(cut != cudaSuccess)
		{
		printf("%s : File: %s  Line: %d \n",cudaGetErrorString(cut),file,line);
		return -1 ;
		}
	return 0;  
}

void matrixProduct(int n, int* a, int* b ,int *erg)
{
	for (int i = 0 ; i < n ; ++i)
	{
		
		for(int j = 0 ; j < n; ++j)
		{
			int scalarProduct = 0; 
			for(int k = 0 ; k < n ; ++k)
			{
				//sclarproduct of i'th row and j'th collumn 
				int scalarProduct += a[i*n+k]* b[k*n+j] ;
			}
			
			erg[i*n+j]= scalarProduct;
		}
		
	}
	
	
}



void diadicProduct(int n, int *a , int *b, int *erg)   // erg = a * b
{
	for(int i=0; i < n; ++i)
		{
			
			for ( int j = 0; j < n; ++j)
				{
					
					erg[i*n+j] = a[i]*b[j]; 
					
				}
		
		}
	
	

}

void printVector(int *vector, int n)
{
	for(int i = 0 ; i < n; ++i)
		{
			printf("  %d \n", vector[i]);	
					
		}
}

void printMatrix(int *matrix,int n)
{
		// print matrix a 
	for(int i = 0; i < n; ++i)
		{
		for (int j = 0 ; j < n; ++j)
			{
				printf("%d",matrix[n*i+j]);
			}	
			printf("\n");
		}
 }






int main(int argc, char**args)
{
	if(argc != 2) 
		{
			printf("Call Programm with program ./programm.out <dimension>");
			return -1 ;
		}
	
	int n = atoi(args[1]) ;
	time_t t;
	int *a = (int *) malloc( sizeof(int)*n*n);
	int *b = (int *) malloc( sizeof(int)*n*n);
	int *c = (int *) malloc( sizeof(int)*n*n);
	int *c_t = (int *) malloc( sizeof(int)*n*n);

	int *a_d = NULL ;
	int *b_d = NULL ;
	int *c_d = NULL ;


	HANDLE_ERROR(cudaMalloc(&a_d, sizeof(int)*n)); // malloc of a_device
	HANDLE_ERROR(cudaMalloc(&b_d, sizeof(int)*n)); // malloc of b_device
	HANDLE_ERROR(cudaMalloc(&c_d, sizeof(int)*n*n)); // malloc of c_device 
	double time1=0.0, tstart;      // time measurment variables
	// random init 
	time(&t);
	srand((unsigned int)t);   


	for(int i = 0; i < n; ++i)
		{
			for (int j= 0; j < n ; ++j)
			{
					a[i*n+j] = rand() % 5 ;	
					b[i*n+j]= rand() % 5 ; 		
			}	
		
		}

	//Transfer a_host to a_device
	HANDLE_ERROR(cudaMemcpy(a_d, a, sizeof(int)*n*n, cudaMemcpyHostToDevice));
		
	//Transfer b_host to b_device
	HANDLE_ERROR(cudaMemcpy(b_d, b, sizeof(int)*n, cudaMemcpyHostToDevice));
	

	

	printf("=============MATRIX A =============  \n");
	printVector(a,n);



	printf("===============MATRIX B================== \n");
	printVector(b,n);

	
	
	printf("====== Result of matrix multiplication =====\n");
	tstart = clock();              // start 
	diadicProduct(n,a,b,c);
	time1 = clock() - tstart;     // end
	time1 = time1/CLOCKS_PER_SEC;  // rescale to seconds
	

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	dim3 block(16,16,1);
	dim3 grid(ceil((float)n/(float)16),ceil((float)n/(float)16),1);
	diadic_Product<<<grid,block>>>(n,a_d,b_d,c_d);
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaMemcpy(c_t, c_d, sizeof(int)*n*n, cudaMemcpyDeviceToHost));
	printf("====== Result of matrix multiplication on Kernel=====\n");
	

	for(int i = 0; i < (n*n); ++i)
		{
			if(c[i] != c_t[i])
				{
					printf("failure at %d",i);
					break; 
				}
		}

	HANDLE_ERROR(cudaFree(a_d));
	HANDLE_ERROR(cudaFree(b_d));
	HANDLE_ERROR(cudaFree(c_d));
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel-diadicProduct: %f msec\n", time );
	printf ("Time for the CPU -diadicProduct %d msec \n",time1 *1000);

	return 0;  	

}
