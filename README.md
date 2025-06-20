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


## 3. Jogo da Vida - CUDA
### Descrição da Implementação

Nesta etapa do projeto, foi desenvolvida a versão CUDA do Jogo da Vida, denominada `jogodavida.cu`, com o objetivo de executar a evolução da sociedade de organismos vivos utilizando uma GPU do cluster chococino.

#### Estrutura de Paralelização
- Cada thread CUDA foi responsável por calcular a evolução de uma célula individual.
- O cálculo da nova geração foi implementado em um kernel CUDA chamado `UmaVidaKernel`.
- A execução paralela foi organizada com blocos de 16x16 threads e grids dimensionados para cobrir todo o tabuleiro.

#### Troca de Dados
- Foi realizada a alocação de memória na GPU com `cudaMalloc`.
- As matrizes do tabuleiro foram transferidas da CPU para a GPU usando `cudaMemcpy`.
- As trocas entre as matrizes de entrada e saída foram realizadas diretamente na GPU, com sincronização entre as chamadas do kernel.

#### Dificuldades e Soluções
- **Desafio:** Gerenciar os índices e a borda do tabuleiro na GPU.
- **Solução:** Foi criada uma função auxiliar `device_ind2d` para o cálculo correto dos índices no código CUDA.
- **Desafio:** Sincronizar corretamente as gerações.
- **Solução:** Utilização de `cudaDeviceSynchronize` após cada chamada do kernel para garantir que as threads terminaram antes da próxima iteração.

#### Código CUDA Desenvolvido
O código desenvolvido está disponível no arquivo `jogodavida.cu` e segue a lógica fornecida no código base sequencial, adaptando a função `UmaVida` para um kernel CUDA com paralelização eficiente.

### Descrição do Experimento

#### Configuração de Teste
- **Host:** 164.41.20.252 
- **Dimensões dos tabuleiros:** Testes realizados para tamanhos 2^3, 2^4, ..., 2^10

#### Procedimento
1. O tempo de execução foi medido para cada tamanho de tabuleiro, utilizando a função `wall_time`.
2. Foram executadas todas as iterações de acordo com o código base, até que o "veleiro" alcançasse o canto inferior direito.
3. As execuções foram feitas no cluster chococino, garantindo compatibilidade com o ambiente de avaliação.

#### Resultados (Exemplo)
| Tamanho (N) | Tempo Total (s) |
|-------------|-----------------|
| 8           | 0.1579599       |
| 16          | 0.0005600       |
| 32          | 0.0010312       |
| 64          | 0.0020881       |
| 128         | 0.0045030       |
| 256         | 0.0113790       |
| 512         | 0.0453660       |
| 1024        | 0.3059671       |


### Conclusão

A implementação CUDA apresentou execução correta, com o "veleiro" alcançando a posição esperada no tabuleiro em todas as execuções. O paralelismo oferecido pela GPU trouxe uma melhora significativa no tempo de execução comparado às versões sequenciais.

Resultados preliminares mostram que quanto maior o tamanho do tabuleiro, maior o benefício da paralelização com CUDA, evidenciando o potencial das GPUs para esse tipo de problema.

Além disso, o código foi desenvolvido com foco na portabilidade e compatibilidade com o cluster chococino, utilizando práticas corretas de alocação, cópia de dados e sincronização.

### Instruções de Compilação e Execução

#### Compilação:
```bash
nvcc -o jogodavida jogodavida.cu
```

#### Execução:
```bash
./jogodavida
```

### Comentários 
- O código foi validado para diferentes tamanhos de tabuleiro.
- A paralelização foi eficiente e a sincronização adequada.
- O código pode ser facilmente ajustado para diferentes configurações de GPU e dimensões de bloco.




