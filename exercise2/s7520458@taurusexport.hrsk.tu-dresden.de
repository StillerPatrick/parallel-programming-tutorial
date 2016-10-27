#include<stdio.h>
#include<assert.h>
#include<math.h>
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


//================== CUDA FUNCTIONS ====================

__global__ void random_init(int n, double *x, double *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate current Thread
	
	if (i < n)
	{
	x[i] = (double)(((unsigned long long int)950706376*i) % 0x7FFFFFFFlu);
	y[i] = (double)(((unsigned long long int)870706376*i) % 0x7FFFFFFFlu);
	}
	
}




// calculate points that are in the circle 

__global__ void calculate_n(int n, double *x, double*y, int *num_points)
{

int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate current Thread

if(i < n)
	{
		if(sqrt((double)(x[i]*x[i]) + (double)(y[i]*y[i])) <= (double)0x7FFFFFFFlu)
			{
				atomicAdd(num_points,1);	
			}
	}

}


int main(int argc, char* argv[])
{
	// ============= INIT =====================

	int sum = 0; 
	double *random_points_x_d = NULL;
	double *random_points_y_d = NULL;
	int *num_points_d = NULL;
	float pi =0; 

	 	

	//============TRANSFER======================
	HANDLE_ERROR(cudaMalloc(&random_points_x_d, sizeof(double)*N)); // malloc of x_device
	HANDLE_ERROR(cudaMalloc(&random_points_y_d , sizeof(double)*N)); // malloc of y_device

	HANDLE_ERROR(cudaMalloc(&num_points_d, sizeof(int))); //malloc of n_points
	HANDLE_ERROR(cudaMemcpy(num_points_d, &sum, sizeof(int), cudaMemcpyHostToDevice));

	
	//=============Calculation ==================
	

	random_init<<<ceil((float)N/(float)256),256>>>(N,random_points_x_d, random_points_y_d);
	calculate_n<<<ceil((float)N/(float)256),256>>>(N,random_points_x_d, random_points_y_d, num_points_d);




	//===========CHECK============================
 
	
	HANDLE_ERROR(cudaMemcpy(&sum,num_points_d, sizeof(int), cudaMemcpyDeviceToHost));
	pi = (float)4 * ((float)sum / (float) N) ;

	printf("%d \n", sum);
	printf("%f \n",pi);


//=========CLEAN==============================

	HANDLE_ERROR(cudaFree(random_points_x_d));
	HANDLE_ERROR(cudaFree(random_points_y_d));
	HANDLE_ERROR(cudaFree(num_points_d));

	return 0 ; 

}

