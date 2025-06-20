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


## MPI

No Jogo da Vida, o desafio computacional é calcular, geração após geração, o estado de cada célula em um tabuleiro que pode ser muito grande. Um único processador precisa percorrer célula por célula, linha por linha, o que se torna extremamente lento para tabuleiros massivos.

A Versão MPI tem como objetivo dividir o trabalho de simulação da evolução da sociedade de organismo vivos entre os processos, o intuito é distribuir o trabalho de modo que ele possa ser realizado mais rápido.

- Como Executar o código MPI

`mpicc -o jogodavidampi jogodavidampi.c` 

`mpirun -np 4 ./jogodavidampi `
