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
 * 	Borwein Paralelo com POSIX Threads
 *
 *	Este programa utiliza o algoritmo Borwein de forma paralela
 *	para calcular o "pi" com precisao de 10 milhoes de casas decimais.
 * 
 */

#include <stdio.h>
#include <gmp.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

//tempo
time_t begin, end;
double time_spent; 
//threads
pthread_t t1, t2;
//variaveis
mpf_t a, y, pi, r, aux1, aux2, pi2;
unsigned int i, pot2 = 2;
pthread_t vetorThreads[50];


//calcula Yi
void *getY1(void *arg){
	//aux1 = 1 - r
	mpf_ui_sub(aux1, 1, r);
}

void *getY2(void *arg){
	//aux1 = 1 + r
	mpf_add_ui(aux2, r, 1);
}

//calcula Ai
void *getA1(void *arg){
	//aux1 = a*((1+y)^4)
	mpf_add_ui(aux1, y, 1);
	mpf_pow_ui(aux1, aux1, 4);
	mpf_mul(aux1, a, aux1);
}

void *getA2(void *arg){
	//aux2 = 2^(2k+3)*y*(1+y+y^2)
	pot2 = pot2 * 4;	
	mpf_mul(aux2, y, y);
	mpf_add(aux2, y, aux2);
	mpf_add_ui(aux2, aux2, 1);
	mpf_mul_ui(aux2, aux2, pot2);
	mpf_mul(aux2, aux2, y);
}

//grava no arquivo o numero pi
void *savePi(void *arg){
	//arquivo
	FILE *fp;
	char filename[20];

	//grava no arquivo
	sprintf(filename, "borwein_paralelo_%d.txt", i);
	fp = fopen(filename, "w");
	mpf_out_str(fp, 10, 0, pi);
	fclose(fp);
}

//inicializa as variaveis
void initAll(){
	mpf_init(a);
	mpf_init(y);
	mpf_init(pi);
	mpf_init(pi2);
	mpf_init(r);
	mpf_init(aux1);
	mpf_init(aux2);
}

//limpa as variaveis
void clearAll(){
	mpf_clear(a);
	mpf_clear(y);
	mpf_clear(pi2);
	mpf_clear(pi);
	mpf_clear(r);
	mpf_clear(aux1);
	mpf_clear(aux2);
}

//com threads e com gmp
void borwein1(){
	time(&begin);
	
	//inicializa e define a precisao de 10000000 * log2(10)
	mpf_set_default_prec(10000100*(log(10)/log(2)));
	initAll();

	//formula a0 = 6 - (4 * sqrt(2))
	mpf_sqrt_ui(a, 2);
	mpf_mul_ui(a, a, 4);
	mpf_ui_sub(a, 6, a);

	//formula y0 = sqrt(2) - 1
	mpf_sqrt_ui(y, 2);
	mpf_sub_ui(y, y, 1);

	//pi inicial = 1/a
	mpf_ui_div(pi, 1, a);

	i=0;
	do{
		//formula r = (1-y^4)^1/4)
		mpf_pow_ui(r, y, 4);
		mpf_ui_sub(r, 1, r);
		mpf_sqrt(r, r);
		mpf_sqrt(r, r);
		//formula aux1 = 1-r
		pthread_create(&t1, NULL, getY1, NULL);
		//formula aux2 = 1+r
		pthread_create(&t2, NULL, getY2, NULL);
		pthread_join(t1, NULL);
		pthread_join(t2, NULL);
		//calcula y = aux1/aux2
		mpf_div(y, aux1, aux2);

		//formula aux1 = a*((1+y)^4)
		pthread_create(&t1, NULL, getA1, NULL);	
		//formula aux2 = (2^(2*k+3))*y*(1+y+y^2)
		pthread_create(&t2, NULL, getA2, NULL);
		pthread_join(t1, NULL);
		pthread_join(t2, NULL);
		//calcula a = aux1 - aux2
		mpf_sub(a, aux1, aux2);

		//pi2 = pi
		mpf_set(pi2, pi);
		//formula a = aux1-aux2
		mpf_ui_div(pi, 1, a);

		pthread_create(&vetorThreads[i], NULL , savePi, NULL); 
		i++;	
	} while(mpf_cmp(pi2, pi)<0);

	int j;
	for(j=0;j<i;j++){
		pthread_join(vetorThreads[j], NULL); 
	}
	
	//printf("iteracoes: %d\n", i);
	
	//libera as variaveis
	clearAll();

	//tempo
	time(&end);
	time_spent = difftime(end, begin);
    	//printf("time : %lf secs\n", time_spent);
	
	printf("Tempo de execucao: %lf segundos\nIteracoes: %d\n", time_spent, i);
}

int main(int argc, char *argv[]){
	borwein1();
	
	return 0;
}
