#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define POWMIN 3
#define POWMAX 10

int tam;
#define ind2d(i,j) ((i)*(tam+2)+(j))

// Função para medir o tempo de execução (em segundos)
double wall_time(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

/*
o código foi modificado do Jogo da Vida tradicional com paralelismo OpenMP, com o diferencial de utilizar recursos de GPU
por meio da diretiva `#pragma omp target`. Para isso, utilizou-se o `#pragma omp target` e `target data` para enviar e manter dados na GPU,
além do `teams distribute parallel for` para executar as iterações em múltiplas threads e equipes na GPU. Com relação ao Mapeamento de memória,
a diretiva `map(...)` foi usada para controlar quais dados são copiados da CPU para GPU e vice-versa. A hierarquia de execução consiste em:
-`target` que define a execução na GPU
-`teams` que é os grupos de threads (como blocos no CUDA)
-`parallel for collapse(2)` que divide dois loops aninhados entre as threads
Esta versão permite o aproveitamento da aceleração por GPU quando suportada.
 */


// Função que calcula uma geração do Jogo da Vida
void UmaVida(int* tabulIn, int* tabulOut, int tam) {
  // Offload da computação para a GPU com OpenMP
  // 'target' envia o bloco para o dispositivo (GPU)
  // 'teams distribute parallel for' divide o trabalho entre múltiplas equipes e threads
  // 'collapse(2)' colapsa dois laços aninhados em um único para melhor distribuição
  // 'map(to: ...)' copia os dados de entrada para a GPU
  // 'map(from: ...)' copia os dados de saída da GPU para a CPU
  #pragma omp target teams distribute parallel for collapse(2) \
    map(to: tabulIn[0:(tam+2)*(tam+2)]) \
    map(from: tabulOut[0:(tam+2)*(tam+2)])
  for (int i = 1; i <= tam; i++) {
    for (int j = 1; j <= tam; j++) {
      int vizviv = tabulIn[ind2d(i-1,j-1)] + tabulIn[ind2d(i-1,j  )] +
                   tabulIn[ind2d(i-1,j+1)] + tabulIn[ind2d(i  ,j-1)] +
                   tabulIn[ind2d(i  ,j+1)] + tabulIn[ind2d(i+1,j-1)] +
                   tabulIn[ind2d(i+1,j  )] + tabulIn[ind2d(i+1,j+1)];

      // Regras do Jogo da Vida
      if (tabulIn[ind2d(i,j)] && vizviv < 2)
        tabulOut[ind2d(i,j)] = 0;           // Solidão
      else if (tabulIn[ind2d(i,j)] && vizviv > 3)
        tabulOut[ind2d(i,j)] = 0;           // Superpopulação
      else if (!tabulIn[ind2d(i,j)] && vizviv == 3)
        tabulOut[ind2d(i,j)] = 1;           // Reprodução
      else
        tabulOut[ind2d(i,j)] = tabulIn[ind2d(i,j)]; // Permanece igual
    }
  }
}

// Inicializa as matrizes de entrada e saída com uma configuração padrão
void InitTabul(int* tabulIn, int* tabulOut, int tam) {
  int ij;
  for (ij = 0; ij < (tam+2)*(tam+2); ij++) {
    tabulIn[ij] = 0;
    tabulOut[ij] = 0;
  }

  // Configuração de um glider (estrutura que se move)
  tabulIn[ind2d(1,2)] = 1; tabulIn[ind2d(2,3)] = 1;
  tabulIn[ind2d(3,1)] = 1; tabulIn[ind2d(3,2)] = 1;
  tabulIn[ind2d(3,3)] = 1;
}

// Verifica se a configuração final está correta (glider deslocado)
int Correto(int* tabul, int tam) {
  int ij, cnt = 0;
  for (ij = 0; ij < (tam+2)*(tam+2); ij++)
    cnt += tabul[ij];

  return (cnt == 5 &&
          tabul[ind2d(tam-2,tam-1)] &&
          tabul[ind2d(tam-1,tam  )] &&
          tabul[ind2d(tam  ,tam-2)] &&
          tabul[ind2d(tam  ,tam-1)] &&
          tabul[ind2d(tam  ,tam  )]);
}

int main(void) {
  int pow, i;
  int *tabulIn, *tabulOut;
  double t0, t1, t2, t3;

  for (pow = POWMIN; pow <= POWMAX; pow++) {
    tam = 1 << pow; // 2^pow

    t0 = wall_time();
    tabulIn  = (int *) malloc((tam+2)*(tam+2)*sizeof(int));
    tabulOut = (int *) malloc((tam+2)*(tam+2)*sizeof(int));
    InitTabul(tabulIn, tabulOut, tam);
    t1 = wall_time();

    // Define região de dados para offload — evita recarregamento de dados a cada iteração
    #pragma omp target data map(tofrom: tabulIn[0:(tam+2)*(tam+2)], \
                                      tabulOut[0:(tam+2)*(tam+2)])
    {
      // Simula várias gerações do jogo alternando buffers
      for (i = 0; i < 2*(tam-3); i++) {
        UmaVida(tabulIn, tabulOut, tam);
        UmaVida(tabulOut, tabulIn, tam);
      }
    }

    t2 = wall_time();

    // Verifica se a configuração esperada foi alcançada
    if (Correto(tabulIn, tam))
      printf("**RESULTADO CORRETO**\n");
    else
      printf("**RESULTADO ERRADO**\n");

    t3 = wall_time();
    printf("tam=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f\n",
           tam, t1-t0, t2-t1, t3-t2, t3-t0);

    free(tabulIn);
    free(tabulOut);
  }

  return 0;
}
