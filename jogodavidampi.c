#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sys/time.h>
#include <mpi.h>

#define ind2d(i,j) (i)*(tam+2)+j
#define POWMIN 3
#define POWMAX 10
#define MASTER 0

double wall_time(void);
void InitTabul(int* tabulIn, int* tabulOut, int tam);
int Correto(int* tabul, int tam);


/* Principal modificação para a versão MPI: Esse função replíca o tabuleiro em versões menores para que cada instância 
do processo trabalhe com uma parte do tabuleiro.
A função UmaVidaLocal calcula a próxima geração do jogo da vida para uma parte do tabuleiro.
Ela recebe o tabuleiro de entrada (tabulIn), o tabuleiro de saída (tabulOut), a ideia é que ele 
receba apenas a parte do tabuleiro que lhe foi atribuída, a partir do que o 
processo mestre (rank 0) enviou para ele, e calcula a próxima geração de células vivas.
A função também considera as células vizinhas, que podem estar em outros processos, e usa
MPI_Sendrecv para trocar as bordas do tabuleiro entre os processos vizinhos, garantindo
que cada processo tenha as informações necessárias para calcular a próxima geração corretamente.*/

void UmaVidaLocal(int* tabulIn, int* tabulOut, int tam, int linhas_locais) {
  int i, j, vizviv;
  for (i = 1; i <= linhas_locais; i++) {
    for (j = 1; j <= tam; j++) {
      vizviv =  tabulIn[ind2d(i-1,j-1)] + tabulIn[ind2d(i-1,j  )] +
                tabulIn[ind2d(i-1,j+1)] + tabulIn[ind2d(i  ,j-1)] +
                tabulIn[ind2d(i  ,j+1)] + tabulIn[ind2d(i+1,j-1)] +
                tabulIn[ind2d(i+1,j  )] + tabulIn[ind2d(i+1,j+1)];
      if (tabulIn[ind2d(i,j)] && vizviv < 2)
        tabulOut[ind2d(i,j)] = 0;
      else if (tabulIn[ind2d(i,j)] && vizviv > 3)
        tabulOut[ind2d(i,j)] = 0;
      else if (!tabulIn[ind2d(i,j)] && vizviv == 3)
        tabulOut[ind2d(i,j)] = 1;
      else
        tabulOut[ind2d(i,j)] = tabulIn[ind2d(i,j)];
    }
  }
}

int main(void) {
  int pow, i, tam;
  int *tabulIn, *tabulOut, *tabul_temp;
  double t0, t1, t2, t3;

  int rank, size;
  int vizinho_cima, vizinho_baixo;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (pow = POWMIN; pow <= POWMAX; pow++) {
    tam = 1 << pow;

    int linhas_por_proc = tam / size;
    int resto = tam % size;
    int linhas_locais = (rank < resto) ? linhas_por_proc + 1 : linhas_por_proc;
    
    // Alocação dos buffers locais
    size_t local_buffer_size = (linhas_locais + 2) * (tam + 2) * sizeof(int);
    int* tabulInLocal = (int*)malloc(local_buffer_size);
    int* tabulOutLocal = (int*)malloc(local_buffer_size);

    // Inicializar os buffers locais com zero em todos os processos.
    // Isso garante que as bordas que não recebem dados (topo do proc 0, base do proc N-1)
    // e o buffer de saída estejam limpos antes do uso.
    memset(tabulInLocal, 0, local_buffer_size);
    memset(tabulOutLocal, 0, local_buffer_size);

    if (rank == MASTER) {
      t0 = wall_time();
      tabulIn  = (int *) malloc ((tam+2)*(tam+2)*sizeof(int));
      tabulOut = (int *) malloc ((tam+2)*(tam+2)*sizeof(int));
      InitTabul(tabulIn, tabulOut, tam);
      t1 = wall_time();
    }

    // Preparação para o Scatterv
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    int current_displ = 0;
    for (i = 0; i < size; ++i) {
        int linhas = (i < resto) ? linhas_por_proc + 1 : linhas_por_proc;
        sendcounts[i] = linhas * (tam + 2);
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }
    
    // Distribuição dos dados do mestre (rank 0) para os buffers locais de todos
    MPI_Scatterv(
      // buffer de envio: começa na primeira linha lógica do tabuleiro global
      (rank == MASTER) ? tabulIn + (tam+2) : NULL, 
      sendcounts, 
      displs, 
      MPI_INT,
      // buffer de recebimento: começa na primeira linha lógica do tabuleiro local
      tabulInLocal + (tam+2), 
      sendcounts[rank], 
      MPI_INT,
      0, 
      MPI_COMM_WORLD);

    if (rank == MASTER) {
      t2 = wall_time();
    }

    vizinho_cima = (rank == MASTER) ? MPI_PROC_NULL : rank - 1;
    vizinho_baixo = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // O laço agora simula duas gerações por iteração, como no original.
    for (i = 0; i < 2 * (tam - 3); i++) {
        // --- Início da 1ª Geração da Iteração ---
        // Troca de fronteiras para o cálculo de tabulInLocal -> tabulOutLocal
        MPI_Sendrecv(tabulInLocal + linhas_locais*(tam+2), tam+2, MPI_INT, vizinho_baixo, 0,
                     tabulInLocal, tam+2, MPI_INT, vizinho_cima, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(tabulInLocal + 1*(tam+2), tam+2, MPI_INT, vizinho_cima, 1,
                     tabulInLocal + (linhas_locais+1)*(tam+2), tam+2, MPI_INT, vizinho_baixo, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Calcula a 1ª geração
        UmaVidaLocal(tabulInLocal, tabulOutLocal, tam, linhas_locais);
        // --- Fim da 1ª Geração da Iteração ---


        // --- Início da 2ª Geração da Iteração ---
        // Troca de fronteiras para o cálculo de tabulOutLocal -> tabulInLocal
        MPI_Sendrecv(tabulOutLocal + linhas_locais*(tam+2), tam+2, MPI_INT, vizinho_baixo, 0,
                     tabulOutLocal, tam+2, MPI_INT, vizinho_cima, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(tabulOutLocal + 1*(tam+2), tam+2, MPI_INT, vizinho_cima, 1,
                     tabulOutLocal + (linhas_locais+1)*(tam+2), tam+2, MPI_INT, vizinho_baixo, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Calcula a 2ª geração (resultado vai para tabulInLocal, que era o buffer antigo)
        UmaVidaLocal(tabulOutLocal, tabulInLocal, tam, linhas_locais);
        // --- Fim da 2ª Geração da Iteração ---
    }
    
    MPI_Barrier(MPI_COMM_WORLD); 
    double t_comp_end = wall_time();


    // Coleta dos resultados finais
    MPI_Gatherv(
      // buffer de envio: resultado final está em tabulInLocal
      tabulInLocal + (tam+2), sendcounts[rank], MPI_INT,
      // buffer de recebimento: no rank 0, montar o resultado em tabulIn
      (rank == MASTER) ? tabulIn + (tam+2) : NULL, sendcounts, displs, MPI_INT,
      0, MPI_COMM_WORLD);

    if (rank == MASTER) {
      t3 = wall_time();
      if (Correto(tabulIn, tam))
        printf("**RESULTADO CORRETO**\n");
      else
        printf("**RESULTADO ERRADO**\n");

      double t_comp = t_comp_end - t2;
      double t_fim = t3 - t_comp_end;
      double t_tot = (t1-t0) + t_comp + t_fim;
      printf("tam=%d; ranks=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n",
             tam, size, t1-t0, t_comp, t_fim, t_tot);
      
      free(tabulIn);
      free(tabulOut);
    }

    free(tabulInLocal);
    free(tabulOutLocal);
    free(sendcounts);
    free(displs);
  }

  MPI_Finalize();
  return 0;
}

// Função para medir o tempo de execução
double wall_time(void) {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return(tv.tv_sec + tv.tv_usec/1000000.0);
}

void InitTabul(int* tabulIn, int* tabulOut, int tam){
  int ij;
  for (ij=0; ij<(tam+2)*(tam+2); ij++) {
    tabulIn[ij] = 0;
    tabulOut[ij] = 0;
  }
  tabulIn[ind2d(1,2)] = 1; tabulIn[ind2d(2,3)] = 1;
  tabulIn[ind2d(3,1)] = 1; tabulIn[ind2d(3,2)] = 1;
  tabulIn[ind2d(3,3)] = 1;
}

int Correto(int* tabul, int tam){
  int ij, cnt = 0;
  for (ij=0; ij<(tam+2)*(tam+2); ij++)
    cnt = cnt + tabul[ij];
  return (cnt == 5 && tabul[ind2d(tam-2,tam-1)] &&
      tabul[ind2d(tam-1,tam  )] && tabul[ind2d(tam  ,tam-2)] &&
      tabul[ind2d(tam  ,tam-1)] && tabul[ind2d(tam  ,tam  )]);
}