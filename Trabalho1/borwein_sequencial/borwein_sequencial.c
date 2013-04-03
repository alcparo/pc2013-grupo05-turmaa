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
 * 	Borwein Sequencial
 *
 *	Este programa utiliza o algoritmo Borwein de forma sequencial
 *	para calcular o "pi" com precisao de 10 milhoes de casas decimais.
 * 
 */

#include <stdio.h>
#include <math.h>
#include <gmp.h>
#include <time.h>

//tempo
time_t begin, end;
double time_spent; 
//variaveis compartilhadas
mpf_t a, y, pi, pi2;
unsigned int i, pot2 = 2;
//arquivo
FILE *fp;
char filename[20];

//calculo y = 1-r/1+r, onde r = (1-(y^4))^1/4
void getY(){
	mpf_t r, aux1;

	mpf_init(r);
	mpf_init(aux1);

	//r = (1-(y^4))^1/4
	mpf_pow_ui(r, y, 4);
	mpf_ui_sub(r, 1, r);
	mpf_sqrt(r, r);
	mpf_sqrt(r, r);

	//calcula y = 1-r/1+r
	mpf_ui_sub(aux1, 1, r);
	mpf_add_ui(r, r, 1);
	mpf_div(y, aux1, r);

	mpf_clear(r);
	mpf_clear(aux1);
}

//calcula a = a*((1+y)^4) - 2^(2k+3)*y*(1+y+y^2)
void getA(){
	mpf_t s, aux1;

	mpf_init(s);
	mpf_init(aux1);

	//s = a*((1+y)^4)
	mpf_add_ui(s, y, 1);
	mpf_pow_ui(s, s, 4);
	mpf_mul(s, a, s);

	//aux1 = 2^(2k+3)*y*(1+y+y^2)
	pot2 = pot2 * 4;	
	mpf_mul(aux1, y, y);
	mpf_add(aux1, y, aux1);
	mpf_add_ui(aux1, aux1, 1);
	mpf_mul_ui(aux1, aux1, pot2);
	mpf_mul(aux1, aux1, y);
	
	//calcula a = s - aux1
	mpf_sub(a, s, aux1);

	mpf_clear(s);
	mpf_clear(aux1);
}

//com threads e com gmp
void borwein1(){
	//inicializa
	mpf_set_default_prec(10000100*(log(10)/log(2)));
	mpf_init(a);
	mpf_init(y);
	mpf_init(pi);
	mpf_init(pi2);

	//tempo inicial
	time(&begin);

	//calcula a0 = 6 - 4*(2^(1/2))
	mpf_sqrt_ui(a, 2);
	mpf_mul_ui(a, a, 4);
	mpf_ui_sub(a, 6, a);

	//calcula y0 = 2^(1/2) - 1
	mpf_sqrt_ui(y, 2);
	mpf_sub_ui(y, y, 1);

	//pi inicial = 1/a
	mpf_ui_div(pi, 1, a);
	
	i=0;
	//calcula ai's e yi's
	do{
		getY();
		getA();
		//armazena pi antigo
		mpf_set(pi2, pi);
		//calcula novo pi
		mpf_ui_div(pi, 1, a);

		//grava no arquivo
		sprintf(filename, "borwein_sequencial_%d.txt", i);
		fp = fopen(filename, "w");
		mpf_out_str(fp, 10, 0, pi);
		close(fp);

		i++;		
	} while(mpf_cmp(pi2, pi)<0);

	//printf("iteracoes : %d\n", i);

	//limpa
	mpf_clear(pi2);
	mpf_clear(a);
	mpf_clear(y);
	mpf_clear(pi);

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
