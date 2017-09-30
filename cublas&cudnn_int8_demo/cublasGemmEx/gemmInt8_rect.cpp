#include <iostream>
using namespace std;

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// cuda
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N       4
#define Value   2
#define checkCudaAPIErrors(F) if ((F) != cudaSuccess) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }

void initArray(char * a, const int size) {
    for (int i = 0; i < size; ++i) 
	{
		a[i] = Value;
    }
}
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";

		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";

		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "<unknown>";
}

#define checkcuBlasError(F) if ((F) != CUBLAS_STATUS_SUCCESS) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, _cudaGetErrorEnum(F)); exit(-1); }

/** @main function ****************
**********************************/
int main(int argc, char** argv)
{
    // test_count
    int iters = 1;

    int alpha = 1;
    int beta  = 0;

    float TFlops;
	cublasStatus_t cublasStat;

    int n[N] = {512,  512,  512,  512};
    int k[N] = {2048, 2048, 2048, 2048};
    int m[N] = {4,    8,    16,   32};

    int devID = 0;
    cudaSetDevice(devID);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devID);
    printf("Device : %s, compute SM %d.%d.\n",devProp.name, devProp.major, devProp.minor);

	cublasHandle_t handle;
	checkcuBlasError(cublasCreate(&handle));

    FILE *output = NULL;
    char filename[20] = "result.txt";

	cudaEvent_t start, stop;
	float time_used = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	char *d_A, *d_B; 
	int  *d_C;	// note the result is accumulated in int
	char *h_A, *h_B;
	int  *h_C;	// note the result is accumulated in int


	if ((output = fopen(filename, "w")) == NULL)
	{
		printf("Can not open file : %s\n", filename);
		exit(1);
	}
    fprintf(output, "m      \t k      \t n      \t Time      \t TFlops\n");

    for (int i=0; i<N; i++)
    {
        // allocate memory
        h_A = (char*)malloc(sizeof(char) * m[i] * k[i]);
        h_B = (char*)malloc(sizeof(char) * k[i] * n[i]);
        h_C = (int *)malloc(sizeof(int ) * m[i] * n[i]);

        checkCudaAPIErrors(cudaMalloc((void **)&d_A, sizeof(char) * m[i] * k[i]));
        checkCudaAPIErrors(cudaMalloc((void **)&d_B, sizeof(char) * k[i] * n[i]));
        checkCudaAPIErrors(cudaMalloc((void **)&d_C, sizeof(int ) * m[i] * n[i]));

        // initilize data on host
        initArray(h_A, m[i] * k[i]);    // init the matrix to 1
        initArray(h_B, k[i] * n[i]);    // init the matrix to 1

        printf("h_A[0] = %x, h_A[last] = %x \n", h_A[0], h_A[m[i]*k[i]-1]);
        printf("h_B[0] = %x, h_B[last] = %x \n", h_B[0], h_B[k[i]*n[i]-1]);

        // copy date from host to device
        checkCudaAPIErrors(cudaMemcpy(d_A, h_A, sizeof(char) * m[i] * k[i],cudaMemcpyHostToDevice));
        checkCudaAPIErrors(cudaMemcpy(d_B, h_B, sizeof(char) * k[i] * n[i],cudaMemcpyHostToDevice));

        // gpu warm up
        cublasStat=cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], 
                    &alpha, d_B, CUDA_R_8I, n[i], d_A, CUDA_R_8I, k[i], &beta, d_C, CUDA_R_32I, n[i],
					CUDA_R_32I,				// specify the computatioin type for cublasGemmEx
					CUBLAS_GEMM_DFALT);		// specify the algorithm for cublasGemmEx
					//CUBLAS_GEMM_ALGO2);		// specify the algorithm for cublasGemmEx
		checkcuBlasError(cublasStat);

        cudaEventRecord(start, 0);

        for (int t = 0; t < iters; t++)
        {
        cublasStat=cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n[i], m[i], k[i], 
                    &alpha, d_B, CUDA_R_8I, n[i], d_A, CUDA_R_8I, k[i], &beta, d_C, CUDA_R_32I, n[i],
					CUDA_R_32I,				// specify the computatioin type for cublasGemmEx
					CUBLAS_GEMM_DFALT);		// specify the algorithm for cublasGemmEx
					//CUBLAS_GEMM_ALGO2);		// specify the algorithm for cublasGemmEx

		checkcuBlasError(cublasStat);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_used, start, stop);
        cout << "cublas sgemm elapsed time: " << time_used / iters << " ms" << endl;

        checkCudaAPIErrors(cudaMemcpy(h_C, d_C, sizeof(int) * m[i] * n[i],cudaMemcpyDeviceToHost));

        // verify the result computed on GPU with int8
        for (int ii=0; ii<m[i]*n[i]; ii++)
        {
            if (h_C[ii] != k[i]*Value*Value)
            {
                printf("get error result on GPU with m = %d, n = %d, C[] = %d\n", m[i], n[i], h_C[ii]);
                break;
            }
        }

        time_used /= (iters);
        TFlops     = (long(2))*m[i]*n[i]*k[i]/(time_used * 1000 * 1000 * 1000); // unit: Tflops
        fprintf(output, "%6d\t%6d\t%6d\t%10.6f\t%10.6f\n", m[i], k[i], n[i], time_used/iters, TFlops);

        checkCudaAPIErrors(cudaMemcpy(h_C, d_C, sizeof(int) * m[i] * n[i],cudaMemcpyDeviceToHost));
        // free memory
        free(h_A);
        free(h_B);
        free(h_C);
        checkCudaAPIErrors(cudaFree(d_A));
        checkCudaAPIErrors(cudaFree(d_B));
        checkCudaAPIErrors(cudaFree(d_C));
    }

	cublasDestroy(handle);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    fclose(output);

	return 0;
}

