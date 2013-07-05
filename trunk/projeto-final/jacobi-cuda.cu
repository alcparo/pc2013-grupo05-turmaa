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
 * 	Projeto Final - Metodo Jacobi-Richardson em CUDA
 *
 *	Este programa resolve sistema lineares utilizando o método de 
 * 	Jacobi-Richardson. O método é implementado de forma paralela 
 * 	utilizando a plataforma CUDA.
 * 
 * 	Ele retorna as seguintes informações:
 * 		-Quantidades de iterações;
 * 		-Comparação entre o resultado utilizando o valor obtido
 * 		aplicado em uma linha de testes com o valor real.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define TPB 128

/* Declarações de parâmetros */
void printMA(int J_ORDER, float *MA); // Função para debug. Imprime a matrix MA.
void printMB(int J_ORDER, float *MB); // Função para debug. Imprime o vetor MB.
void printX(int J_ORDER, float *X); // Função para debug. Imprime o vetor X.
void printResults(int ite, int J_ROW_TEST, float result, float MB); // Apresenta os resultados formatados.

/*	__global__ void diag(float *dev_MA, float *dev_MB, float *dev_X, int J_ORDER);
 * 
 * 	Descricao:
 * 		Esta função realiza o cálculo da matriz MA* e do vetor MB*, além de atribuir
 * 		o valor inicial para o vetor X. O retorno da função se dá em modificar os próprios
 * 		parâmetros de entrada.
 * 		Cada linha da matriz MA é calculada em um bloco e cada coluna dessa linha é calculada
 * 		pelas threads desse bloco.
 * 
 * 	Parametros de entrada:
 * 		-(float) dev_MA: Matriz MA alocada da memória da GPU.
 * 		-(float) dev_MB: Vetor MB alocado na memória da GPU.
 * 		-(float) dev_X: Vetor X alocado na memória da GPU.
 * 		-(int) J_ORDER: Ordem dos vetores.
 * 
 * 	Parametros de saida:
 * 		- 
 * 		
 */
__global__ void diag(float *dev_MA, float *dev_MB, float *dev_X, int J_ORDER){

	int bid = blockIdx.x; 	// Id do bloco.
	int tid = threadIdx.x; 	// Id da thread.
	
	/* Condição para evitar erros. Impede que blocos e threads executem o código fora do nosso controle. */
	if(bid < J_ORDER && tid < J_ORDER){
		
		/* Declaração e atribuição da variável auxiliar que irá armazenar o valor da diagonal na linha "bid" */
		int diagAux; 		
		diagAux = dev_MA[bid*J_ORDER+bid];
		
		/* Para cada linha calculada em um bloco (bId), as threads desse bloco irão calcular o valor de cada coluna */
		while(tid<J_ORDER){
			
			if(bid!=tid){
				dev_MA[tid*J_ORDER+bid] = dev_MA[tid*J_ORDER+bid]/diagAux;
			} else{
				dev_MA[tid*J_ORDER+bid] = 0;
			}		
			
			/* Incremento o tId para que todas as colunas sejam calculadas, já que cada bloco não executa mais que 512 threads */
			tid += TPB;
		}
		
		/* Espero todas as threads do block bId executarem os seus cálculos e atribuo a saída nos parâmetros de entrada */
		__syncthreads();
		dev_MB[bid] = dev_MB[bid] / diagAux;
		dev_X[bid] = dev_MB[bid];			
	}	
}

/*	__global__ void jacobi(float *dev_MA, float *dev_X, int J_ORDER, float *dev_sum);
 * 
 *   	Descricao:
 * 		Esta função realiza o cálculo do somatório presente no algoritmo. Para isso, utilizamos 
 * 		a seguinte abordagem: cada linha a ser multiplicada e somada é controlada por um bloco (bId),
 * 		cada uma das colunas dessa linha é controlada por uma thread (tId). Cada thread irá realizar
 * 		o cálculo das colunas de posição tId e tId+TPB (threads per block). Assim, caso a matriz tenha
 * 		mais que 512 colunas, todas elas serão calculadas. O somatório de cada thread será salvo numa
 * 		posição do cache compartilhado. Por fim, realizamos uma redução desse cache compartilhado, salvando
 * 		o resultado final (o somatório da linha), na sua posição referente em dev_sum.
 * 
 * 	Parametros de entrada:
 * 		-(float) dev_MA: Matriz MA alocada da memória da GPU.
 * 		-(float) dev_MB: Vetor MB alocado na memória da GPU.
 * 		-(float) dev_X: Vetor X alocado na memória da GPU.
 * 		-(int) J_ORDER: Ordem dos vetores.
 * 		-(float) dev_sum: Vetor que armazenará os somatórios
 * 
 * 	Parametros de saida:
 * 		-
 * 
 */
__global__ void jacobi(float *dev_MA, float *dev_X, int J_ORDER, float *dev_sum){
	
	int bId = blockIdx.x; // Id do bloco.
	int tId = threadIdx.x; // Id da thread.
	int cacheIndex = threadIdx.x; // Indice do cache da thread.
	float temp = 0; // Variável auxiliar que irá armazenar o somatório de cada thread.
	
	__shared__ float cache[TPB]; // Cache compartilhado que irá armazenar o somatório de cada thread, utilizado na posterior redução.
	cache[cacheIndex] = 0; // Atribuição inicial.
	
	/* Condição para evitar erros. Impede que blocos e threads executem o código fora do nosso controle. */
	if(bId < J_ORDER && tId < J_ORDER){
		
		/* Para cada linha (bId) da Matriz, as threads desse bloco irão seus respectivos somatórios */
		while(tId < J_ORDER){
			if(bId!=tId){
				temp += (dev_MA[(tId * J_ORDER) + bId] * dev_X[tId]);
			}
			
			tId += TPB;
		}
		
		/* Salvo o somatório de cada thread na sua posição respectiva do cache e aguardo todas as threads terminarem os seus cálculos */
		cache[cacheIndex] = temp;
		__syncthreads();
		
		/* Redução em paralelo */
		int i = TPB/2;
		while(i != 0){
			if(cacheIndex < i )
				cache[cacheIndex] += cache[cacheIndex + i];
			
			__syncthreads();
			i /= 2;
		}
		
		/* A thread 0 irá realizar a atribuição do valor final. */
		if(cacheIndex == 0){
			dev_sum[bId] = cache[0];			
		}
	}
}

/*	int main(int argc, char * argv[]);
 * 
 * 	Descricao:
 * 		Na main é realizada a leitura do arquivo de entrada bem como a declaração, alocação de memória
 * 		das variáveis envolvidas e a comunidação entre host e device.
 * 		Primeiramente o kernel diag é chamado para realizar o cálculo de MA* e MB*, bem como a atribuição inicial de X.
 * 		Então, uma estrutura de repetição irá realizar chamadas ao kernel jacobi para que o somatório seja realizado na gpu.
 * 		O somatório é copiado para o host e a atribuição do novo valor de X é feita, bem como o cálculo do erro. A estrutura 
 * 		de repetição para se ela atingir o número máximo de iterações ou o erro mínimo.
 * 
 * 	Parametros de entrada:
 * 		- argc: Quantidade de parametros de entrada junto a execucao do programa.
 * 		- argv: Array contendo strings com cada parametro de entrada.
 * 
 * 	Parametros de saida:
 * 		- (int): Retorna 0 caso o programa seja executado com sucesso. 
 */
int main(int argc, char * argv[]){
			
	//ENTRADA
	FILE * pFile; // Variavel para a abertura do arquivo de entrada.
	/* Abertura do arquivo de entrada */
	if((argc < 2) || (pFile = fopen(argv[1],"r")) == NULL){
		fputs ("File error",stderr); exit (1);
	}
	
	int i,j; // Variavel de controle para as estrutudas de repeticao.
	
	/* Parâmetros de entrada */
	int J_ORDER, J_ROW_TEST;
	float J_ERROR, J_ITE_MAX;	

	fscanf(pFile, "%d%d%f%f", &J_ORDER, &J_ROW_TEST, &J_ERROR, &J_ITE_MAX);	
	
	/* Declaração e alocamento das variáveis do host */
	float *MA, *MB, *X, *Xold, *MAOriginal, MBOriginal, *sum;		
	MA = 		(float*)malloc(sizeof(float)*J_ORDER*J_ORDER);
	MAOriginal = 	(float*)malloc(sizeof(float)*J_ORDER);	
	MB = 		(float*)malloc(sizeof(float)*J_ORDER);
	X = 		(float*)malloc(sizeof(float)*J_ORDER);
	Xold = 		(float*)malloc(sizeof(float)*J_ORDER);
	sum = 		(float*)malloc(sizeof(float)*J_ORDER);
	
	/* Leitura do arquivo de entrada */
	for(i = 0; i<J_ORDER; i++){		
		for(j = 0; j<J_ORDER; j++){
			fscanf(pFile, "%f", &MA[j*J_ORDER+i]);
			if(i==J_ROW_TEST)
				MAOriginal[j] = MA[j*J_ORDER+i];
		}
	}	
		
	for(i=0; i<J_ORDER; i++){
		fscanf(pFile, "%f", &MB[i]);
		if(i == J_ROW_TEST)
			MBOriginal = MB[i];		
	}
	
	fclose(pFile);

	/* Declaração e alocamento das variáveis do device */
	float *dev_MA, *dev_MB, *dev_X, *dev_sum;	
	cudaMalloc( (void**)&dev_MA, J_ORDER*J_ORDER*sizeof(float));
	cudaMalloc( (void**)&dev_MB, J_ORDER*sizeof(float));
	cudaMalloc( (void**)&dev_X, J_ORDER*sizeof(float));
	cudaMalloc( (void**)&dev_sum, J_ORDER*sizeof(float));
	
	/* Cópia de valores do host para o device */
	cudaMemcpy(dev_MA, MA, J_ORDER*J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MB, MB, J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_X, X, J_ORDER*sizeof(float), cudaMemcpyHostToDevice);

	/* Chamada ao kernel diag
	* Blocos: J_ORDER - cada bloco cuida de uma linha
	* Threads: TPB - cada thread cuida de uma colunas		 * 
	*/
	diag<<<J_ORDER , TPB>>>(dev_MA, dev_MB, dev_X, J_ORDER); // OK
	
	/* Cópia dos valores de MB* e X do device para o host */
	cudaMemcpy(MB, dev_MB, J_ORDER*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(X, dev_X, J_ORDER*sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Variáveis auxiliáres para o cálculo de X */
	int ite = 0;
	float maxDif, maxX, Mr;	
	Mr = FLT_MAX;
		
	/* Estrutura de repetição que irá realizar chamadas ao quernel jacobi, calcular o novo valor de X e o erro */
	while(ite < J_ITE_MAX && Mr > J_ERROR){
		
		/* Chamada ao kernel jacobi
		 * Blocos: J_ORDER - cada bloco cuida de uma linha
		 * Threads: TPB - cada thread cuida de uma colunas		 * 
		 */
		jacobi<<<J_ORDER , TPB>>>(dev_MA, dev_X, J_ORDER, dev_sum);
		
		/* Cópia dos valores do somatório do device para o host */
		cudaMemcpy(sum, dev_sum, J_ORDER*sizeof(float), cudaMemcpyDeviceToHost);
		
		/* Cálculo do novo X e do erro */
		maxDif = maxX = FLT_MIN;
		for(i = 0 ; i < J_ORDER; i++){
			Xold[i] = X[i];
			X[i] = (MB[i] - sum[i]); // Novo X
			
			if(fabs(X[i] - Xold[i]) > maxDif)
				maxDif = fabs(X[i] - Xold[i]);
				
			if(fabs(X[i]) > maxX)
				maxX = fabs(X[i]);
		}
		
		Mr = maxDif / maxX; // Erro
		
		ite++; // Iteração
		
		/* Cópia do novo X para o device */
		cudaMemcpy(dev_X, X, J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
		
	}
	
	/* Cálculo do resultado utilizando a linha de teste */
	float resultAux=0;
	for(j=0; j<J_ORDER; j++){
		resultAux += MAOriginal[j]*X[j];
	}
	
	/* Impressão do resultado formatado */
	printResults(ite, J_ROW_TEST, resultAux, MBOriginal);
	
	/* Liberação de memória */
	free(MA); free(MB); free(X); free(Xold); free(MAOriginal); free(sum);
	cudaFree(dev_MA); cudaFree(dev_MB); cudaFree(dev_X); cudaFree(dev_sum);
	
	return 0;
	
}


/*	void printMA(int J_ORDER, float *MA);
 *
 *   	Descricao:
 * 		Impressão da matriz MA para debug. 
 * 
 * 	Parametros de entrada:
 * 		-(int) J_ORDER: Ordem da matriz.
 * 		-(float) MA: matriz MA.
 * 
 * 	Parametros de saida: 
 * 		- 
 */		
void printMA(int J_ORDER, float *MA){
	int i,j;
	for(i = 0; i<750; i++){
		for(j = 0; j<750; j++){
			printf("%f ", MA[j*J_ORDER+i]);
		}
		printf("\n");
	}	
}

/*	void printMB(int J_ORDER, float *MB);
 *
 *   	Descricao:
 * 		Impressão do vator MB para debug. 
 * 
 * 	Parametros de entrada:
 * 		-(int) J_ORDER: Ordem dos vetores.
 * 		-(float) MB: Vetor MB.
 * 
 * 	Parametros de saida: 
 * 		- 
 */	
void printMB(int J_ORDER, float *MB){
	int i;
	for(i = 0; i<J_ORDER; i++){		
		printf("%f ", MB[i]);		
		printf("\n");
	}	
}

/*	void printX(int J_ORDER, float *X);
 *
 *   	Descricao:
 * 		Impressão do vator X para debug. 
 * 
 * 	Parametros de entrada:
 * 		-(int) J_ORDER: Ordem dos vetores.
 * 		-(float) X: Vetor X.
 * 
 * 	Parametros de saida: 
 * 		-
 */	
void printX(int J_ORDER, float *X){
	int i;
	for(i = 0; i<J_ORDER; i++){		
		printf("%f ", X[i]);		
		
	}	
	printf("\n");
}

/*	void printResults(int ite, int J_ROW_TEST, float result, float MB)
 * 
 *   	Descricao:
 * 		Impressão formatada dos resultados.
 * 
 * 	Parametros de entrada:
 * 		-(int) ite: Número de iterações obtido.
 * 		-(int) J_ROW_TEST: Linha de teste.
 * 		-(float) result: Resultado obtido.
 * 		-(float) MB: Resultado esperado.
 * 
 * 	Parametros de saida: 
 * 		-
 */
void printResults(int ite, int J_ROW_TEST, float result, float MB){
	printf("\n\n---------------------------------------------------------\n"
	"Iterations: %d\n"
	"RowTest: %d => [%f] =? %f\n"
	"---------------------------------------------------------\n\n", ite, J_ROW_TEST, result, MB);
}