/**
 * 	Instituto de Ciencias Matematicas e de Computacao - USP Sao Carlos
 * 
 * 	Programacao Concorrente 2013
 *	Grupo 05 Turma A
 *
 *	Andre Luiz Catini Paro, 	7152740 
 * 	Daniel Hideki Yoshimi,		7239173
 * 	Rodrigo Toledo Amancio Silva,	7152308
 * 
 * 	Gauss-Legendre Paralelo com POSIX Threads
 *
 *	Este programa utiliza o algoritmo Gauss-Legendre de forma paralela
 *	para calcular o "pi" com precisao de 10 milhoes de casas decimais.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

  /* Declaracao das funcoes utilizadas */
  void initAttributes();
  void *calculate_a(void *args);
  void *calculate_b(void *args);
  void *calculate_t(void *args);
  void *calculate_p(void *args);
  void *calculate_pi(void *args);
  void *filePrint(void *args);

  /* Declaracao das variaveis globais utilizadas */
  long int n_count; 			//Variavel para armazenar a iteracao atual do algoritmo
  mpf_t *a_n, *b_n, *t_n, *p_n, *pi;	//Variaveis que irao armazenar os valores das variaveis 
					//do algoritmo a cada iteracao      
  pthread_t t_calculate_a, t_calculate_b, t_calculate_p, t_calculate_t, t_calculate_pi, t_filePrint[30];	//Variaveis que armazenarao as threads irao calcular os valores das variaveis 
					//do algoritmo a cada iteracao   

/**
* int main()
* 
* Descricao:
* 	Funcao principal do programa que contem um loop (do-while) contendo a chamada
* 	das threads que calculam as variaveis utilizadas pelo algoritmo Gauss-Legendre
* 	a cada iteracao. 
* 
* Parametros de entrada:
* 	-
* 
* Parametros de retorno:
*	-
*	
*/
int main(){
  
  /* Variaveis utilizadas para calcular o tempo de execucao do programa */
  time_t begin, end;
  double time_spent;
  time(&begin);
  
  /* Inicialicazao das variaveis globais utilizadas no algoritmo */
  initAttributes();  
  
  /* Loop principal que calcula o valor do "pi" */
  do{
    //printf("Iteracao: %ld\n", n_count); 
    
    pthread_create(&t_calculate_a, NULL, calculate_a, NULL);
    pthread_create(&t_calculate_b, NULL, calculate_b, NULL);
    pthread_create(&t_calculate_p, NULL, calculate_p, NULL); 
    pthread_create(&t_calculate_t, NULL, calculate_t, NULL); 
    pthread_create(&t_calculate_pi, NULL, calculate_pi, NULL);       
    pthread_join(t_calculate_pi, NULL);        
    pthread_create(&t_filePrint[n_count], NULL, filePrint, (void *)n_count); 
    pthread_join(t_calculate_p, NULL);
    
    n_count++;
  } while(mpf_cmp(pi[n_count-2], pi[n_count-1])!=0); 
    
  int i;  
  for(i=0; i<n_count; i++){
    pthread_join(t_filePrint[i], NULL); 
  }  
  
  time(&end);
  time_spent = difftime(end, begin);	

  printf("Tempo de execucao: %lf segundos\nIteracoes: %ld\n", time_spent, n_count);
  
  return 0;
}

/**
 * void *filePrint(void *args)
 * 
 * Descricao:
 * 	Imprime a cada iteracao o valor do "pi" num arquivo de nome "gauss_legendre_sequencial_X.txt", onde
 * 	X é o numero da iteracao, comecando em 1. Recebe como parametro o valor n_count da iteracao atual
 * 	para que ele possa imprimir os valores do "pi" em arquivos diferentes de forma paralela conforme os
 * 	valores vao sendo calculados.
 * 
 * Parametros de entrada:
 * 	void *args: Argumento necessario para a execucao da funcao em pthreads. Tambem recebe o valor n_count
 * 	da iteracao atual. 
 * 
 * Parametros de retorno:
 * 	void *: Retorna NULL. Argumento necessario para a execucao da funcao em pthreads.
 */
void *filePrint(void *args){
  
  char filename[30];  
  
  long int n_count_print;
  n_count_print = (long int) args;
  
  sprintf(filename, "gauss_legendre_paralelo_%ld.txt", n_count_print+1);
  FILE *output = fopen(filename, "w");  
  mpf_out_str(output, 10, 0, pi[n_count_print]);
  fclose(output);  
  
  return NULL;
}

/**
 * void initAttributes()
 * 
 * Descricao:
 * 	Inicializa as variaveis globas e atribui os valores iniciais necessarios para a execucao do algoritmo
 * 	Gauss-Legendre.
 * 
 * Parametros de entrada:
 * 	-
 * 
 * Parametros de retorno:
 * 	-
 */
void initAttributes(){
  
  /* Atribui como precisao default 10 milhoes de casas decimais. A funcao mpf_set_default_prec() tem como parametro de entrada
   * a precisao desejada em bits, ou seja, e necessario utilizar a conversao (log de 2 na base 10)*10 milhoes para obter o valor correto. */
  mpf_set_default_prec((log(10)/log(2))*10000000);

  int i;
  mpf_t sqrt2; 	// Variavel auxiliar para o calculo de b0    
  mpf_t b0;  	// Variavel auxiliar para o calculo de b0
  
  n_count = 0;

  /* Arrays de variaveis utilizadas pelo algoritmo. Como o algoritmo converge para o valor correto do "pi" com 45 milhoes de casas decimais
   * em apenas 25 iteracoes, serao utilizado arrays com 25 posicoes */
  a_n = (mpf_t*) malloc(25*sizeof(mpf_t));
  b_n = (mpf_t*) malloc(25*sizeof(mpf_t));
  t_n = (mpf_t*) malloc(25*sizeof(mpf_t));
  p_n = (mpf_t*) malloc(25*sizeof(mpf_t));
  pi = (mpf_t*) malloc(25*sizeof(mpf_t));  
  
  for(i=0; i<25; i++){
    mpf_init(a_n[i]);
    mpf_init(b_n[i]);
    mpf_init(p_n[i]);
    mpf_init(t_n[i]);
    mpf_init(pi[i]);    
  }
  
  /* Atribuicao dos valores iniciais */
  mpf_init(sqrt2);
  mpf_init(b0);
  mpf_sqrt_ui(sqrt2, 2);
  mpf_ui_div(b0, 1, sqrt2);  
  
  mpf_set_d(a_n[n_count], 1.0);
  mpf_set(b_n[n_count], b0);
  mpf_set_d(t_n[n_count], 0.25);
  mpf_set_d(p_n[n_count], 1.0);
  
}

/**
 * void *calculate_a(void *args)
 * 
 * Descricao:
 * 	Calcula o valor da variavel a na n-esima iteracao.
 * 
 * Parametros de entrada:
 * 	void *args: Argumento necessario para a execucao da funcao em pthreads.
 * 
 * Parametros de retorno:
 * 	void *: Retorna NULL. Argumento necessario para a execucao da funcao em pthreads
 */
void *calculate_a(void *args){
  
    mpf_add(a_n[n_count+1], a_n[n_count], b_n[n_count]);
    mpf_div_ui(a_n[n_count+1], a_n[n_count+1], 2);   
    
    return NULL;
}

/**
 * void *calculate_b(void *args)
 * 
 * Descricao:
 * 	Calcula o valor da variavel b na n-esima iteracao.
 * 
 * Parametros de entrada:
 * 	void *args: Argumento necessario para a execucao da funcao em pthreads.
 * 
 * Parametros de retorno:
 * 	void *: Retorna NULL. Argumento necessario para a execucao da funcao em pthreads
 */
void *calculate_b(void *args){

    mpf_mul(b_n[n_count+1], a_n[n_count], b_n[n_count]);
    mpf_sqrt(b_n[n_count+1], b_n[n_count+1]);
    
    return NULL;  
}

/**
 * void *calculate_t(void *args)
 * 
 * Descricao:
 * 	Calcula o valor da variavel t na n-esima iteracao.
 * 
 * Parametros de entrada:
 * 	void *args: Argumento necessario para a execucao da funcao em pthreads.
 * 
 * Parametros de retorno:
 * 	void *: Retorna NULL. Argumento necessario para a execucao da funcao em pthreads
 */
void *calculate_t(void *args){

    pthread_join(t_calculate_a, NULL); 
    mpf_sub(t_n[n_count+1], a_n[n_count], a_n[n_count+1]);
    mpf_pow_ui(t_n[n_count+1], t_n[n_count+1], 2);    
    mpf_mul(t_n[n_count+1], p_n[n_count], t_n[n_count+1]);
    mpf_sub(t_n[n_count+1], t_n[n_count], t_n[n_count+1]);
  
    return NULL;
}

/**
 * void *calculate_p(void *args)
 * 
 * Descricao:
 * 	Calcula o valor da variavel p na n-esima iteracao.
 * 
 * Parametros de entrada:
 * 	void *args: Argumento necessario para a execucao da funcao em pthreads.
 * 
 * Parametros de retorno:
 * 	void *: Retorna NULL. Argumento necessario para a execucao da funcao em pthreads
 */
void *calculate_p(void *args){

    mpf_mul_ui(p_n[n_count+1], p_n[n_count], 2);  
    return NULL;  
}

/**
 * void *calculate_pi(void *args)
 * 
 * Descricao:
 * 	Calcula o valor do "pi" na n-esima iteracao.
 * 
 * Parametros de entrada:
 * 	void *args: Argumento necessario para a execucao da funcao em pthreads.
 * 
 * Parametros de retorno:
 * 	void *: Retorna NULL. Argumento necessario para a execucao da funcao em pthreads
 */
void *calculate_pi(void *args){ 
    
    pthread_join(t_calculate_b, NULL);
    mpf_add(pi[n_count], a_n[n_count+1], b_n[n_count+1]);
    mpf_pow_ui(pi[n_count], pi[n_count], 2);    
    mpf_div_ui(pi[n_count], pi[n_count], 4);
    pthread_join(t_calculate_t, NULL);
    mpf_div(pi[n_count], pi[n_count], t_n[n_count+1]); 
    
    return NULL;  
}
