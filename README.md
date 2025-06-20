# PSPD-jogo-da-vida
Trabalho Extraclasse da disciplina PSPD, chamado Jogo da Vida,

Aluno | Matricula
--|--
Artur Vinicius Dias Nunes | 190142421
Henrique Hida | 180113569
João Manoel Barreto Neto | 211039519 
Leonardo Milomes Vitoriano | 201000379
Miguel Matos Costa de Frias Barbosa | 211039635


[Enunciado](PSPD_LAB_JOGO_DA_VIDA.pdf)


## 1. MPI

No Jogo da Vida, o desafio computacional é calcular, geração após geração, o estado de cada célula em um tabuleiro que pode ser muito grande. Um único processador precisa percorrer célula por célula, linha por linha, o que se torna extremamente lento para tabuleiros massivos.

A Versão MPI tem como objetivo dividir o trabalho de simulação da evolução da sociedade de organismo vivos entre os processos, o intuito é distribuir o trabalho de modo que ele possa ser realizado mais rápido.

- Como compilar o código MPI:

`mpicc -o jogodavidampi.bin jogodavidampi.c` 

- Como executar o código MPI:

`mpirun -np 4 ./jogodavidampi `


## 2. OMP (OpenMP)


### 2.1 Jogo da Vida com OMP

O código foi otimizado utilizando a biblioteca **OpenMP** para explorar o paralelismo de múltiplos núcleos durante a simulação do Jogo da Vida. A principal função paralelizada é `UmaVida`, que aplica as regras do autômato celular em cada célula do tabuleiro. A diretiva `#pragma omp parallel for` foi utilizada sobre o loop externo que percorre as linhas do tabuleiro. Como cada linha pode ser processada de forma independente (não há dependência entre as células de linhas diferentes), essa paralelização é segura e eficaz. Para garantir a segurança entre threads, **as variáveis internas do loop (j e vizviv) foram declaradas como private**.

Com essa modificação, o tempo de execução da simulação foi significativamente reduzido em tabuleiros grandes (por exemplo, 512x512 ou 1024x1024), aproveitando o poder computacional de CPUs com múltiplos núcleos. A paralelização foi implementada de forma simples, sem a necessidade de reestruturar o algoritmo original, demonstrando a eficácia e facilidade de uso do OpenMP para aplicações com laços paralelizáveis.

### 2.2 Compilar e executar

- Como compilar o código OMP:

`gcc -o jogodavidaomp.bin jogodavidaomp.c -fopenmp` 

- Como executar o código OMP:

`./jogodavidaomp.bin` 


### 2.3 Resultado

O resultado da execução pode ser visualizado na imagem abaixo:

![omp](assets/omp.png)



