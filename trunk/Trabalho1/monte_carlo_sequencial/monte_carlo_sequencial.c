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
 * 	Monte-Carlo Sequencial
 *
 *	Este programa utiliza o algoritmo Monte-Carlo de forma sequencial
 *	para calcular o "pi" com precisao de 10 milhoes de casas decimais.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <gmp.h>
#include <unistd.h>

#define POTENC 9
#define PRECISION 128

int main(void) {
	/* Declaracao de variaveis */
	mpf_t ptsCirc;
	mpf_t px,py,val;
	gmp_randstate_t rnd;
	unsigned long i=0,ini=0, amostras=0;
	
	/* Inicializacao de variaveis*/
	mpf_init(ptsCirc);
	mpf_set_default_prec(PRECISION);
	mpf_init(py);
	mpf_init(val);
	mpf_init(px);

	amostras=pow(10,POTENC);

	gmp_randinit_mt(rnd);
	gmp_randseed_ui(rnd,time (NULL));
	ini=time(NULL);	

	/* Loop principal do algoritmo */
	for(i = 0; i<amostras; i++) {
		mpf_urandomb(px,rnd,PRECISION);
		mpf_urandomb(py,rnd,PRECISION);    

		mpf_mul(px,px,px);
		mpf_mul(py,py,py);
		mpf_add(px,py,px);

		if(mpf_cmp_d(px,1.0)<=0){
			mpf_add_ui(ptsCirc,ptsCirc,1);
		}
		
  
		/* Realiza o calculo do PI */
		if(i % 10000000==0 && i>0){
			mpf_set_ui(py,i);
			mpf_div(px,ptsCirc,py);
			mpf_mul_ui(px,px,4);

			/* Imprime na tela o calculo do PI. A impressao na tela foi escolhida pois
			 * o algoritmo fornece numeros com poucas casas decimais de precisao */
			char *output;
			mp_exp_t exp;
			output = mpf_get_str(NULL, &exp, 10, 0, px);
	    		printf("PI = %.*s.%s\n", (int)exp, output, output+exp);
		}
	}

	/* Calculo do tempo de execucao */
	ini=time(NULL)-ini;
	//printf("Tempo=%lds %ld interacoes\n",ini,amostras);
	printf("Tempo de execucao: %lds segundos\nIteracoes: %ld\n", ini, amostras);
	
	/* Liberacao de memória */
	mpf_clear(ptsCirc);
	gmp_randclear (rnd);
	mpf_clear(val);
	mpf_clear(px);
	mpf_clear(py);

	
	return 0;
}
