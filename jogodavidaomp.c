#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>


#define ind2d(i,j) (i)*(tam+2)+j
#define POWMIN 3
#define POWMAX 10



/*
===============================================================================
Jogo da Vida de Conway com Paralelismo OpenMP
-------------------------------------------------------------------------------
Este programa simula o autômato celular "Game of Life" para tabuleiros de 
tamanhos crescentes, utilizando a configuração inicial conhecida como "glider".

PARALELIZAÇÃO:
A função UmaVida(), responsável por aplicar as regras do jogo em cada célula 
do tabuleiro, foi paralelizada com OpenMP utilizando a diretiva:
    #pragma omp parallel for

Essa diretiva paraleliza o loop externo (linha por linha do tabuleiro), pois 
as iterações são independentes — cada thread processa uma linha do tabuleiro 
sem interferir nas demais.

As variáveis internas ao loop, como `j` (coluna) e `vizviv` (vizinhos vivos), 
são declaradas como `private` para garantir que cada thread utilize suas 
próprias cópias, evitando condições de corrida.

===============================================================================
*/



double wall_time(void) {
    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);
    return(tv.tv_sec + tv.tv_usec/1000000.0);
}  /* fim-wall_time */



void UmaVida(int* tabulIn, int* tabulOut, int tam) {
    int i, j, vizviv;

    #pragma omp parallel for private(j, vizviv)
    for (i=1; i<=tam; i++) {
        for (j= 1; j<=tam; j++) {
            vizviv = tabulIn[ind2d(i-1,j-1)] + tabulIn[ind2d(i-1,j )] +
            tabulIn[ind2d(i-1,j+1)] + tabulIn[ind2d(i ,j-1)] +
            tabulIn[ind2d(i ,j+1)] + tabulIn[ind2d(i+1,j-1)] +
            tabulIn[ind2d(i+1,j )] + tabulIn[ind2d(i+1,j+1)];
            if (tabulIn[ind2d(i,j)] && vizviv < 2)
                tabulOut[ind2d(i,j)] = 0;
            else if (tabulIn[ind2d(i,j)] && vizviv > 3)
                tabulOut[ind2d(i,j)] = 0;
            else if (!tabulIn[ind2d(i,j)] && vizviv == 3)
                tabulOut[ind2d(i,j)] = 1;
            else
                tabulOut[ind2d(i,j)] = tabulIn[ind2d(i,j)];
        } /* fim-for */
    } /* fim-for */
} /* fim-UmaVida */



// DumpTabul: Imprime trecho do tabuleiro entre
// as posicoes (first,first) e (last,last)
void DumpTabul(int * tabul, int tam, int first, int last, char* msg){
    int i, ij;
    printf("%s; Dump posições [%d:%d, %d:%d] de tabuleiro %d x %d\n", \
    msg, first, last, first, last, tam, tam);
    for (i=first; i<=last; i++) printf("="); printf("=\n");
    for (i=ind2d(first,0); i<=ind2d(last,0); i+=ind2d(1,0)) {
        for (ij=i+first; ij<=i+last; ij++)
        printf("%c", tabul[ij]? 'X' : '.');
        printf("\n");
        }
    for (i=first; i<=last; i++) printf("="); printf("=\n");
} /* fim-DumpTabul */



// InitTabul: Inicializa tabulIn com veleiro no canto superior esquerdo e
// tabulOut com celulas mortas
void InitTabul(int* tabulIn, int* tabulOut, int tam){
    int ij;
    for (ij=0; ij<(tam+2)*(tam+2); ij++) {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    } /* fim-for */

    tabulIn[ind2d(1,2)] = 1; tabulIn[ind2d(2,3)] = 1;
    tabulIn[ind2d(3,1)] = 1; tabulIn[ind2d(3,2)] = 1;
    tabulIn[ind2d(3,3)] = 1;
} /* fim-InitTabul */



// Correto: Verifica se a configuracao final do tabuleiro
// está correta ou não (veleiro no canto inferior direito)
int Correto(int* tabul, int tam){
    int ij, cnt;
    cnt = 0;
    for (ij=0; ij<(tam+2)*(tam+2); ij++)
        cnt = cnt + tabul[ij];

    return (cnt == 5 && tabul[ind2d(tam-2,tam-1)] &&
        tabul[ind2d(tam-1,tam )] && tabul[ind2d(tam ,tam-2)] &&
        tabul[ind2d(tam ,tam-1)] && tabul[ind2d(tam ,tam )]);
} /* fim-Correto */



int main(void) {
    int pow, i, tam, *tabulIn, *tabulOut;
    char msg[9];
    double t0, t1, t2, t3;

    // para todos os tamanhos do tabuleiro
    for (pow=POWMIN; pow<=POWMAX; pow++) {
        
        tam = 1 << pow;
        
        // aloca e inicializa tabuleiros
        t0 = wall_time();
        tabulIn = (int *) malloc ((tam+2)*(tam+2)*sizeof(int));
        tabulOut = (int *) malloc ((tam+2)*(tam+2)*sizeof(int));
        
        InitTabul(tabulIn, tabulOut, tam);
        
        t1 = wall_time();
        
        for (i=0; i<2*(tam-3); i++) {
            UmaVida(tabulIn, tabulOut, tam);
            UmaVida(tabulOut, tabulIn, tam);
        } /* fim-for */
        
        t2 = wall_time();
        
        if (Correto(tabulIn, tam))
            printf("**RESULTADO CORRETO**\n");
        else
            printf("**RESULTADO ERRADO**\n");
        
        t3 = wall_time();
        
        printf("tam=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n", tam, t1-t0, t2-t1, t3-t2, t3-t0);
        
        free(tabulIn); free(tabulOut);
    }

    return 0;
}
