/*
 * Jogo da Vida com paralelismo híbrido: OpenMP + CUDA
 *
 * DIFERENCIAIS DESTA VERSÃO EM RELAÇÃO AO CÓDIGO BASE:
 *
 * - Utiliza CUDA para calcular a evolução da sociedade (núcleo computacional na GPU).
 * - Utiliza OpenMP para:
 *    → Paralelizar a execução de experimentos para diferentes tamanhos de tabuleiro.
 *    → Inicializar os tabuleiros de entrada e saída com threads.
 *    → Verificar a correção do resultado com redução paralela.
 *
 * Esta abordagem tira proveito tanto da CPU quanto da GPU:
 * - CPU paraleliza múltiplos experimentos em diferentes threads.
 * - GPU executa os cálculos internos de evolução celular.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>         // Para medir tempo com gettimeofday
#include <cuda_runtime.h>     // Biblioteca CUDA para gerenciamento de memória e kernels
#include <omp.h>              // Biblioteca OpenMP para paralelismo em CPU

// Definição dos limites de tamanho (2^POWMIN até 2^POWMAX)
#define POWMIN 3
#define POWMAX 10

// Macro para mapear coordenadas 2D em índice 1D (vetor linearizado)
#define ind2d(i,j,tam) ((i)*(tam+2)+(j))

// Kernel CUDA que realiza uma geração do Jogo da Vida
__global__ void UmaVidaKernel(int* tabulIn, int* tabulOut, int tam) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Evita acessar fora dos limites válidos do tabuleiro
    if (i <= tam && j <= tam) {
        int idx = ind2d(i, j, tam);

        // Soma dos vizinhos vivos
        int vizviv = tabulIn[ind2d(i-1,j-1,tam)] + tabulIn[ind2d(i-1,j  ,tam)] +
                     tabulIn[ind2d(i-1,j+1,tam)] + tabulIn[ind2d(i  ,j-1,tam)] +
                     tabulIn[ind2d(i  ,j+1,tam)] + tabulIn[ind2d(i+1,j-1,tam)] +
                     tabulIn[ind2d(i+1,j  ,tam)] + tabulIn[ind2d(i+1,j+1,tam)];

        // Regras do Jogo da Vida
        if (tabulIn[idx] && vizviv < 2)
            tabulOut[idx] = 0; // Morre por solidão
        else if (tabulIn[idx] && vizviv > 3)
            tabulOut[idx] = 0; // Morre por superpopulação
        else if (!tabulIn[idx] && vizviv == 3)
            tabulOut[idx] = 1; // Célula nasce
        else
            tabulOut[idx] = tabulIn[idx]; // Permanece como está
    }
}

// Função que mede o tempo com precisão
double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

// Inicializa os tabuleiros (com um padrão conhecido)
void InitTabul(int* tabulIn, int* tabulOut, int tam) {
    // Paraleliza a limpeza das matrizes
    #pragma omp parallel for
    for (int ij = 0; ij < (tam+2)*(tam+2); ij++) {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    }

    // Insere um padrão de glider (formato fixo)
    tabulIn[(1)*(tam+2)+(2)] = 1;
    tabulIn[(2)*(tam+2)+(3)] = 1;
    tabulIn[(3)*(tam+2)+(1)] = 1;
    tabulIn[(3)*(tam+2)+(2)] = 1;
    tabulIn[(3)*(tam+2)+(3)] = 1;
}

// Verifica se o padrão final está correto
int Correto(int* tabul, int tam) {
    int cnt = 0;

    // Conta o total de células vivas com redução paralela
    #pragma omp parallel for reduction(+:cnt)
    for (int ij = 0; ij < (tam+2)*(tam+2); ij++)
        cnt += tabul[ij];

    // Verifica se o padrão final esperado foi mantido (glider deslocado)
    return (cnt == 5 &&
            tabul[(tam-2)*(tam+2)+(tam-1)] &&
            tabul[(tam-1)*(tam+2)+(tam  )] &&
            tabul[(tam  )*(tam+2)+(tam-2)] &&
            tabul[(tam  )*(tam+2)+(tam-1)] &&
            tabul[(tam  )*(tam+2)+(tam  )]);
}

// Função principal
int main(void) {
    double t0, t1, t2, t3;

    // Testa vários tamanhos de tabuleiro em paralelo (cada thread faz um experimento)
    #pragma omp parallel for schedule(dynamic)
    for (int pow = POWMIN; pow <= POWMAX; pow++) {
        int tam = 1 << pow;  // tam = 2^pow

        int *tabulIn, *tabulOut;
        int *d_tabulIn, *d_tabulOut;
        size_t size = (tam+2)*(tam+2)*sizeof(int);

        t0 = wall_time();

        // Alocação de memória no host
        tabulIn = (int*)malloc(size);
        tabulOut = (int*)malloc(size);
        InitTabul(tabulIn, tabulOut, tam);

        // Alocação de memória na GPU
        cudaMalloc((void**)&d_tabulIn, size);
        cudaMalloc((void**)&d_tabulOut, size);

        // Copia dados do host para GPU
        cudaMemcpy(d_tabulIn, tabulIn, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tabulOut, tabulOut, size, cudaMemcpyHostToDevice);

        t1 = wall_time();

        // Define quantidade de blocos e threads
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((tam+threadsPerBlock.x-1)/threadsPerBlock.x,
                       (tam+threadsPerBlock.y-1)/threadsPerBlock.y);

        // Executa múltiplas gerações (2*(tam-3) evoluções)
        for (int i = 0; i < 2*(tam-3); i++) {
            UmaVidaKernel<<<numBlocks, threadsPerBlock>>>(d_tabulIn, d_tabulOut, tam);
            cudaDeviceSynchronize();
            UmaVidaKernel<<<numBlocks, threadsPerBlock>>>(d_tabulOut, d_tabulIn, tam);
            cudaDeviceSynchronize();
        }

        // Copia o resultado de volta para a CPU
        cudaMemcpy(tabulIn, d_tabulIn, size, cudaMemcpyDeviceToHost);

        t2 = wall_time();

        // Valida o resultado
        int resultado = Correto(tabulIn, tam);

        t3 = wall_time();

        // Impressão dos tempos de execução e se o resultado está correto
        #pragma omp critical
        {
            if (resultado)
                printf("**CORRETO** ");
            else
                printf("**INCORRETO** ");
            printf("tam=%d; init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f\n",
                   tam, t1-t0, t2-t1, t3-t2, t3-t0);
        }

        // Libera memória
        free(tabulIn); free(tabulOut);
        cudaFree(d_tabulIn); cudaFree(d_tabulOut);
    }

    return 0;
}
