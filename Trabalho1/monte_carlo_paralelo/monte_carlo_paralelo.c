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
 * 	Monte-Carlo Paralelo com POSIX Threads
 *
 *	Este programa utiliza o algoritmo Monte-Carlo de forma paralela
 *	para calcular o "pi" com precisao de 10 milhoes de casas decimais.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <gmp.h>
#include <unistd.h>

#define NUMTHR 4
#define POTENC 9
#define PRECISION 128

mpf_t ptsCirc,ptsTot;
gmp_randstate_t rnd;
unsigned long amostras = 0,ini=0,cont2=0;
pthread_spinlock_t spinlock;	//declaração do multiplex para controle das threads


void *monte_carlo(){
	long i;
	mpf_t px,py;

	mpf_init(py);
	mpf_init(px);

	while(mpf_cmp_d(ptsTot,amostras)<0) {
		mpf_urandomb(px,rnd,PRECISION);
		mpf_urandomb(py,rnd,PRECISION);     
	
		mpf_mul(px,px,px);
		mpf_mul(py,py,py);
		mpf_add(px,py,px);

		if(mpf_cmp_d(px,1.0)<=0){
			pthread_spin_lock(&spinlock);		//trava o acesso das outras threads ao que está abaixo
			mpf_add_ui(ptsCirc,ptsCirc,1);
			mpf_add_ui(ptsTot,ptsTot,1);
			cont2++;
			pthread_spin_unlock(&spinlock);		//libera o acesso das outras thread
		}else{
			pthread_spin_lock(&spinlock);
			mpf_add_ui(ptsTot,ptsTot,1);
			cont2++;
			pthread_spin_unlock(&spinlock);
		}
		if(cont2 % 10000000==0){
			mpf_set_ui(py,cont2);
			mpf_div(px,ptsCirc,py);
			mpf_mul_ui(px,px,4);
			char *output;
			mp_exp_t exp;
			output = mpf_get_str(NULL, &exp, 10, 0, px);
	    		printf("PI = %.*s.%s\n", (int)exp, output, output+exp);
		}
	}
	mpf_clear(py);
	mpf_clear(px);
	pthread_exit(0);			//desnecessariamente finaliza a thread (pode ignorar essa linha se quiser) mas é bom colocar
}

int main(void) {
	pthread_t id[NUMTHR]; //declaração do array que conterá as pthreads
	int i;
	mpf_t val,px;
	long ini;

	mpf_init(ptsCirc);
	mpf_init(ptsTot);
	mpf_init(px);
	mpf_init(val);
	mpf_set_default_prec(PRECISION);

	amostras = pow(10,POTENC);
	gmp_randinit_mt(rnd);
	gmp_randseed_ui(rnd,time (NULL));
	ini=time(NULL);	
	//Mutiplex (mais rápido que pthread_multiplex) para controlar concorrencia - inicialização padrão
	pthread_spin_init(&spinlock, 0);

	ini=time(NULL);

	for(i=0;i<NUMTHR;i++){
		//Cria a thread de id=i, com uma função "run"=monte_carlo
		pthread_create(&id[i],NULL,monte_carlo,NULL);
	}

	for(i=0;i<NUMTHR;i++){
		//aguarda a thread id=i terminar para seguir em frente
		pthread_join(id[i],NULL);
	}

	ini=time(NULL)-ini;
	//printf("Tempo=%lds %ld interacoes\n",ini,amostras);
	printf("Tempo de execucao: %lds segundos\nIteracoes: %ld\n", ini, amostras);
//Free**********************************************************************************************
	pthread_spin_destroy(&spinlock);//da um free no multiplex
	mpf_clear(ptsCirc);
	mpf_clear(ptsTot);
	gmp_randclear (rnd);
	mpf_clear(val);
	mpf_clear(px);
//**************************************************************************************************


	return 0;
}



