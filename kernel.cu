#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
cublasHandle_t handle;
void initiate(){
	cublasCreate(&handle);

}
void
matrixMulCPU(double *C, const double *A, const double *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (double)sum;
        }
}
// Device variables
double *d_A,*d_B,*d_C;
double *h_CUBLAS;
void initialise_memory(int maxn,int maxy){
    int mem_size = maxn * maxy * sizeof(double);
    cudaMallocManaged(&d_A,mem_size);
    cudaMallocManaged(&d_B,mem_size);
    // cudaMallocManaged(&d_C,mem_size);

    cudaMalloc((void **) &d_C, mem_size);   
    h_CUBLAS = (double *) malloc(mem_size);
}
void free_memory(){
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);    
    free(h_CUBLAS);
}
template<typename T> T par_mat_mult(const T& h_A,const T& h_B){
    // cudaEvent_t start;
    // cudaEventCreate(&start);

    // cudaEvent_t stop;
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, NULL);

	// Allocating Host Memory
	int size_A = h_A.n * h_A.m, size_B = h_B.n * h_B.m;
	int mem_size_A = size_A * sizeof(double), mem_size_B = size_B * sizeof(double);

    for(int i = 0; i < h_A.n; i++){
    	for(int j = 0;j < h_A.m; j++){
    		d_A[i * h_A.m + j] = h_A[i][j];
    	}
    }
    for(int i = 0; i < h_B.n; i++){
    	for(int j = 0;j < h_B.m; j++){
    		d_B[i * h_B.m + j] = h_B[i][j];
    	}
    }   


	int size_C = h_A.n * h_B.m;
    int mem_size_C = size_C * sizeof(double);
    	


    const double alpha = 1.0;
    const double beta  = 0.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, h_B.m, h_A.n, h_A.m, &alpha, d_B, h_B.m, d_A, h_A.m, &beta, d_C, h_B.m);  




    cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    // matrixMulCPU(d_C,d_A,d_B,h_A.n,h_A.m,h_B.m);
    T result(h_A.n, h_B.m);
    for(int i = 0; i < h_A.n; i++){
    	for(int j = 0;j < h_B.m; j++){
    		result[i][j] = h_CUBLAS[i * h_B.m + j];
    	}
    }

    // cudaEventRecord(stop, NULL);
    // cudaEventSynchronize(stop);
    // float msecTotal = 0.0f;
    // cudaEventElapsedTime(&msecTotal, start, stop);
    // std::cout<<msecTotal<<std::endl;
    
    return result;
}