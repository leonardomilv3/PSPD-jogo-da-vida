#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define ind2d(i,j) ((i)*(tam+2)+(j))
#define POWMIN 3
#define POWMAX 10

__device__ int device_ind2d(int i, int j, int tam) {
    return i * (tam + 2) + j;
}

__global__ void UmaVidaKernel(int* tabulIn, int* tabulOut, int tam) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= tam && j <= tam) {
        int vizviv = tabulIn[device_ind2d(i-1, j-1, tam)] + tabulIn[device_ind2d(i-1, j, tam)] +
                     tabulIn[device_ind2d(i-1, j+1, tam)] + tabulIn[device_ind2d(i, j-1, tam)] +
                     tabulIn[device_ind2d(i, j+1, tam)] + tabulIn[device_ind2d(i+1, j-1, tam)] +
                     tabulIn[device_ind2d(i+1, j, tam)] + tabulIn[device_ind2d(i+1, j+1, tam)];
        
        if (tabulIn[device_ind2d(i, j, tam)] && vizviv < 2)
            tabulOut[device_ind2d(i, j, tam)] = 0;
        else if (tabulIn[device_ind2d(i, j, tam)] && vizviv > 3)
            tabulOut[device_ind2d(i, j, tam)] = 0;
        else if (!tabulIn[device_ind2d(i, j, tam)] && vizviv == 3)
            tabulOut[device_ind2d(i, j, tam)] = 1;
        else
            tabulOut[device_ind2d(i, j, tam)] = tabulIn[device_ind2d(i, j, tam)];
    }
}

double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

void InitTabul(int* tabulIn, int* tabulOut, int tam) {
    int ij;
    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++) {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    }
    tabulIn[ind2d(1, 2)] = 1; tabulIn[ind2d(2, 3)] = 1;
    tabulIn[ind2d(3, 1)] = 1; tabulIn[ind2d(3, 2)] = 1;
    tabulIn[ind2d(3, 3)] = 1;
}

int Correto(int* tabul, int tam) {
    int ij, cnt = 0;
    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++)
        cnt += tabul[ij];
    
    return (cnt == 5 && tabul[ind2d(tam-2, tam-1)] &&
            tabul[ind2d(tam-1, tam)] && tabul[ind2d(tam, tam-2)] &&
            tabul[ind2d(tam, tam-1)] && tabul[ind2d(tam, tam)]);
}

int main(void) {
    int pow, i, tam, *tabulIn, *tabulOut, *d_tabulIn, *d_tabulOut;
    double t0, t1, t2, t3;
    
    for (pow = POWMIN; pow <= POWMAX; pow++) {
        tam = 1 << pow;
        
        size_t size = (tam + 2) * (tam + 2) * sizeof(int);
        
        t0 = wall_time();
        
        tabulIn = (int*)malloc(size);
        tabulOut = (int*)malloc(size);
        InitTabul(tabulIn, tabulOut, tam);
        
        cudaMalloc((void**)&d_tabulIn, size);
        cudaMalloc((void**)&d_tabulOut, size);
        
        cudaMemcpy(d_tabulIn, tabulIn, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tabulOut, tabulOut, size, cudaMemcpyHostToDevice);
        
        t1 = wall_time();
        
        dim3 blockSize(16, 16);
        dim3 gridSize((tam + blockSize.x - 1) / blockSize.x, 
                     (tam + blockSize.y - 1) / blockSize.y);
        
        for (i = 0; i < 2 * (tam - 3); i++) {
            UmaVidaKernel<<<gridSize, blockSize>>>(d_tabulIn, d_tabulOut, tam);
            cudaDeviceSynchronize();
            
            UmaVidaKernel<<<gridSize, blockSize>>>(d_tabulOut, d_tabulIn, tam);
            cudaDeviceSynchronize();
        }
        
        cudaMemcpy(tabulIn, d_tabulIn, size, cudaMemcpyDeviceToHost);
        
        t2 = wall_time();
        
        if (Correto(tabulIn, tam))
            printf("**RESULTADO CORRETO**\n");
        else
            printf("**RESULTADO ERRADO**\n");
        
        t3 = wall_time();
        printf("tam=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n", 
               tam, t1-t0, t2-t1, t3-t2, t3-t0);
        
        free(tabulIn);
        free(tabulOut);
        cudaFree(d_tabulIn);
        cudaFree(d_tabulOut);
    }
    return 0;
} 