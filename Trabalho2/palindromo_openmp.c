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
 * 	Palindromos - Paralelo com openMP
 *
 *	Este programa calcula a quantidade de ocorrencias de palindromos em um texto.
 * 	Para duas entradas pre-definidas, ele retorna:
 * 
 * 	shakespeare.txt
 * 		- Quantidade de palindromos em palavras;
 * 		- Quantidade de palindromos em frases;
 * 		- Tempo de execucao.
 * 
 * 	wikipedia.txt
 * 		- Quantidade de palindromos em palavras;
 * 		- Quantidade de palindromos em frases;
 * 		- Quantidade de numeros primos calculado a partir da soma do codigo ASCII dos palindromos;
 * 		- Tempo de execucao.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Declaração do cabeçalho das funcoes */

int phrasePalindrome(char* buffer, char * argv, int* phrasePrimeCount);
int wordPalindrome(char* buffer, char * argv, int* wordPrimeCount);
char** wordTokenize(const char* str);
char** wordTokenizeDetails(const char* str);
char** phraseTokenize(const char* str);
int crivoEratostenes(int num);

/**
 *  	int main(int argc, char * argv[])
 * 
 * 	Descricao:
 * 		Principal funcao do programa. Realiza a leitura do arquivo de entrada e a chamada das funcoes
 * 		que irao calcular os palindromos.
 * 
 * 	Parametros de entrada:
 * 		- argc: Quantidade de parametros de entrada junto a execucao do programa.
 * 		- argv: Array contendo strings com cada parametro de entrada.
 * 
 * 	Parametros de saida:
 * 		- (int): Retorna 0 caso o programa seja executado com sucesso.
 * 	
 */
int main(int argc, char * argv[]){
		
	
	/* Declaracao de variaveis */
	double timeStart = omp_get_wtime( ); // Armazena o tempo de inicio do programa, em segundos.	
	
	int i; // Variavel de controle para as estrutudas de repeticao.
	int NUMTHREADS; // Variavel que ira armazenar a quantidade de threads utilizadas pelo openMP
	
	FILE * pFile; // Variavel para a abertura do arquivo de entrada.
	long lSize; // Variavel que armazena o tamanho do arquivo de entrada.
	char * buffer; // Variavel que ira armazenar o arquivo de entrada na memoria.
	size_t result; // Variavel que ira armazenar o retorno da funcao de leitura do arquivo de entrada. 
	
	/* Verificacao dos parametros de entrada */
	if(argc < 3 || argv[2] == NULL){ // Caso nao seja especificado o numero de threads a ser utilizado, sera utilizado o maximo possivel.
		NUMTHREADS = omp_get_max_threads();
		omp_set_num_threads(NUMTHREADS);
	} else{ // Caso contrario
		if(atoi(argv[2]) <= omp_get_max_threads()){
			NUMTHREADS = atoi(argv[2]);
			omp_set_num_threads(NUMTHREADS);			
		} else{ // Caso o numero de threads especificado seja maior que o maximo possivel.
			printf("Numero de threads nao suportado. O maximo possivel foi setado.");
			NUMTHREADS = omp_get_max_threads();
			omp_set_num_threads(NUMTHREADS);
		}
	}
	
	/* Abertura do arquivo de entrada */
	if((argc < 2) || (pFile = fopen(argv[1],"r")) == NULL){
		fputs ("File error",stderr); exit (1);
	}
	
	/* Obtencao do tamanho do arquivo de entrada */
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
	
	/* Alocacao de memoria para armazenar o arquivo de entrada */
	buffer = (char*) malloc (sizeof(char)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
	
	/* Copia do arquivo de entrada para a memoria */
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	
	/* Tratamento do arquivo de entrada. Troca de "carriage return" por quebra de linha */
	for(i=0;i<result;i++){ 
		if(buffer[i]=='\r'){
			buffer[i]='\n';  
		}
	}
	
	/* Declaracao de variaveis */
	int bufferBegin[NUMTHREADS], bufferEnd[NUMTHREADS]; // Variaveis que irao armazenar o inicio e fim de cada particao do texto de entrada.
	int bufferSize = (lSize / NUMTHREADS); // Variavel que ira armazena o tamanho inicial de cada particao do arquivo de entrada.
	char* bufferArray[NUMTHREADS]; // Array que ira armazenar cada particao do arquivo de entrada.
	int divided = 0; // Variavel de controle para verificar se o arquivo de entrada terminou de ser particionado.
	
	/* Estrutura de repeticao que realiza o calculo das posicoes de inicio e fim iniciais das particoes do arquivo de entrada */
	bufferBegin[0] = 0; bufferEnd[0] = bufferSize;
	for(i=1; i<NUMTHREADS; i++){
		bufferBegin[i] = bufferEnd[i-1]+1;
		bufferEnd[i] = bufferBegin[i]+bufferSize;
		if(i==NUMTHREADS-1)
			bufferEnd[i]=lSize;
	}
	
	/* Estrutura de repeticao que realiza a particao do arquivo de entrada em n particoes, onde n = NUMTHREADS */
	for(i=0; i<NUMTHREADS-1; i++){
		divided = 0;

		while(!divided){
			if(buffer[bufferEnd[i]] == '!' || buffer[bufferEnd[i]] == '.' || buffer[bufferEnd[i]] == '?'){ // Verifica se o caracter verificado e um delimitador de frase
				bufferArray[i] = strndup(buffer+bufferBegin[i], bufferEnd[i]-bufferBegin[i]+1); // +1  pra pegar o ponto em si
				divided = 1;
			} else{ // Caso nao seja um delimitador de frase, atualiza as posicoes das particoes.
				bufferEnd[i]++;
				bufferBegin[i+1]++;				
			}
		}		
	}
	
	/* Atribuicao da ultima particao */
	bufferArray[NUMTHREADS-1] = strndup(buffer+bufferBegin[NUMTHREADS-1], bufferEnd[NUMTHREADS-1]-bufferBegin[NUMTHREADS-1]+1);
	
	/* Declaracao e inicializacao de variaveis */
	int wordPalindromeCount = 0, phrasePalindromeCount = 0; // Irao armazenar a quantidade total de palindromos por palavra e por frase.
	int wordPalindromeCountAux, phrasePalindromeCountAux; // Variaveis auxiliares para utilizacao do openMP
	
	int* wordPrimeCountAux[NUMTHREADS]; // Array de variaveis auxiliares para utilizacao do openMP
	int* phrasePrimeCountAux[NUMTHREADS]; // Array de variaveis auxiliares para utilizacao do openMP
	int wordPrimeCount = 0, phrasePrimeCount = 0; // Irao armazenar a quantidade total de palindromos de palavra e frase que sao primos
	
	/* Alocacao de memoria para as variaveis auxiliares */
	for(i=0; i<NUMTHREADS; i++){
		wordPrimeCountAux[i] = (int*)malloc(sizeof(int));	*wordPrimeCountAux[i] = 0;
		phrasePrimeCountAux[i] = (int*)malloc(sizeof(int));	*phrasePrimeCountAux[i] = 0;
	}
	
	/**
	 * Secao paralela openMP
	 * 
	 * private(wordPalindromeCountAux, phrasePalindromeCountAux): Variaveis locais de cada secao paralela
	 * reduction(+: wordPalindromeCount, phrasePalindromeCount, wordPrimeCount, phrasePrimeCount): Variaveis que irao armazenar a soma total dos calculos de cada threads
	 * 
	 * Cada thread analiza a particao do texto correspondente ao seu thread_num.
	 * 
	 */
	#pragma omp parallel private(wordPalindromeCountAux, phrasePalindromeCountAux) reduction(+: wordPalindromeCount, phrasePalindromeCount, wordPrimeCount, phrasePrimeCount)
	{
		wordPalindromeCountAux = wordPalindrome(bufferArray[omp_get_thread_num()], argv[1], wordPrimeCountAux[omp_get_thread_num()]); // Calcula a quantidade de palindromos em cada particao do texto
		phrasePalindromeCountAux = phrasePalindrome(bufferArray[omp_get_thread_num()], argv[1], phrasePrimeCountAux[omp_get_thread_num()]); // Calcula a quantidade de palindromos em cada particao do texto

		/* Soma final dos valores calculados em cada thread. Decomposicao por reducao. */
		wordPalindromeCount += wordPalindromeCountAux;
		phrasePalindromeCount += phrasePalindromeCountAux;		
		wordPrimeCount += *wordPrimeCountAux[omp_get_thread_num()];
		phrasePrimeCount += *phrasePrimeCountAux[omp_get_thread_num()];		
	}
	
	/* Impressao dos resultados no terminal */	
	if(!strcmp(argv[1], "shakespe.txt")){		
		printf("%s\nWord palindromes: %d\nPhrase palindromes: %d\n", argv[1], wordPalindromeCount, phrasePalindromeCount);
	} else if(!strcmp(argv[1], "wikipedia.txt")){		
		printf("%s\nWord palindromes: %d\t\tWord prime palindromes: %d\nPhrase palindromes: %d\t\tPhrase prime palindromes: %d\n",
		argv[1], wordPalindromeCount, wordPrimeCount, phrasePalindromeCount, phrasePrimeCount);
	}
	
	/* Liberacao de memoria */
	free(*wordPrimeCountAux);
	free(*phrasePrimeCountAux);
	free(buffer);
	
	/* Impressao do tempo total de execucao */
	double timeEnd = omp_get_wtime( );	
	printf("Execution time: %lf\n",timeEnd-timeStart);
	
	return 0;	
}


/**
 *  	int crivoEratostenes(int num)
 * 
 * 	Descricao:
 * 		Para um determinado numero de entrada, verifica se este e primo ou nao, utilizando a implementacao
 * 		do Crivo de Eratostenes.
 * 
 * 	Parametros de entrada:
 * 		- num: Valor a ser verificado se e primo.
 * 
 * 	Parametros de saida:
 * 		- (int): Retorna 1 caso for primo, 0 caso contrario.
 * 	
 */
int crivoEratostenes(int num){
	
	/* Declaracao de variaveis */
	int primo = 0; // Variavel de retorno.
	int i, j; // Variaveis de controle das estruturas de repeticao.	
	int* vetor = (int*) malloc(sizeof(int)*(num+1)); // Array utilizado para execucao do algoritmo do crivo.
	
	/* Inicializa o array com os numeros de 2 a num */
	for(i=2; i<=num; i++)
		vetor[i] = i;
	
	/* Realiza o calculo do crivo. */
	for(i=2; i<=num; i++){
		if(vetor[i] == i){ // Se TRUE, o numero i e primo.
			
			for(j=i*i; j<=num; j+=i){ // Retira os multiplos de i para a proxima iteracao.
				vetor[j] = 0;
			}
			
			if(i == num){ // Caso o numero primo i seja igual a num, retorna 1 => O valor de entrada e primo.
				primo=1;
				break;
			}
		}
	}

	/* Liberacao de memoria */
	free(vetor);
	
	return primo;
}

/**
 *  	char** phraseTokenize(const char* str)
 * 
 * 	Descricao:
 * 		Funcao que realiza o parser do arquivo de entrada em frases.
 * 		Frase: Trechos de texto terminados por alguma das seguintes pontuacoes: ".", "!" e "?"
 * 		Este codigo, mais otimizado do que simplesmente utilizar "strtok()", foi encontrado no link a seguir e, 
 * 		entao, adaptado para o nosso problema.
 * 		http://stackoverflow.com/questions/8106765/using-strtok-in-c
 * 
 * 	Parametros de entrada:
 * 		- str: String a ser realizado o parser.
 * 
 * 	Parametros de saida:
 * 		- (char**): Ponteiro para o resultado do parser. Cada posicao aponta para o trecho da entrada "parseada".
 * 	
 */
char** phraseTokenize(const char* str)
{
	/* Inicializacao de variaveis */
	int count = 0; // Contador para a quantidade de tokes.
	int capacity = 10; // Tamanho do toke, utilizado para o alocamento de memoria, inicializado inicialmente em 10.
	char** result = malloc(capacity*sizeof(*result)); // Variavel de retorno.	
	const char* e=str; // Variavel auxiliar. 
	
	if (e) do 
	{
		const char* s=e;
		e=strpbrk(s,".!?"); // Delimitadores do parser. Sempre que um desses caracteres for encontrado, sera criado um toke contendo o trecho de texto anterior a ele.
		
		if (count >= capacity)
			result = realloc(result, (capacity*=2)*sizeof(*result));
		
		result[count++] = e? strndup(s, e-s) : strdup(s);
	} while (e && *(++e));
	
	if (count >= capacity)
		result = realloc(result, (capacity+=1)*sizeof(*result));
	result[count++] = 0;
	
	return result;
}

/**
 *  	char** wordTokenize(const char* str)
 * 
 Descricao:
 * 		Funcao que realiza o parser do arquivo de entrada em palavras.
		Palavra: qualquer palavra delimitada por caracteres especiais, menos "-" e "'". (ex: saint-michel e o'clock sao palavras)
 * 		Este codigo, mais otimizado do que simplesmente utilizar "strtok()", foi encontrado no link a seguir e, 
 * 		entao, adaptado para o nosso problema.
 * 		http://stackoverflow.com/questions/8106765/using-strtok-in-c
 * 
 * 	Parametros de entrada:
 * 		- str: String a ser realizado o parser.
 * 
 * 	Parametros de saida:
 * 		- (char**): Ponteiro para o resultado do parser. Cada posicao aponta para o trecho da entrada "parseada".
 * 	
 */
char** wordTokenize(const char* str)
{
	/* Inicializacao de variaveis */
	int count = 0; // Contador para a quantidade de tokes.
	int capacity = 10; // Tamanho do toke, utilizado para o alocamento de memoria, inicializado inicialmente em 10.
	char** result = malloc(capacity*sizeof(*result)); // Variavel de retorno.	
	const char* e=str; // Variavel auxiliar. 
	
	if (e) do 
	{
		const char* s=e;
		e=strpbrk(s," \r\n\t\".,;:(){}[]0123456789!~^><+=\\&*/_#"); // Delimitadores do parser. Sempre que um desses caracteres for encontrado, sera criado um toke contendo o trecho de texto anterior a ele.
		
		if (count >= capacity)
			result = realloc(result, (capacity*=2)*sizeof(*result));
		
		result[count++] = e? strndup(s, e-s) : strdup(s);
	} while (e && *(++e));
	
	if (count >= capacity)
		result = realloc(result, (capacity+=1)*sizeof(*result));
	result[count++] = 0;
	
	return result;
}

/**
 *  	char** wordTokenizeDetails(const char* str)
 * 
 Descricao:
 * 		Funcao que realiza o parser do arquivo de entrada em palavras, de forma mais detalhada. Retira os simbolos anteriormente inclusos nas palavras, "-" e "'". 
 * 		Este codigo, mais otimizado do que simplesmente utilizar "strtok()", foi encontrado no link a seguir e, 
 * 		entao, adaptado para o nosso problema.
 * 		http://stackoverflow.com/questions/8106765/using-strtok-in-c
 * 
 * 	Parametros de entrada:
 * 		- str: String a ser realizado o parser.
 * 
 * 	Parametros de saida:
 * 		- (char**): Ponteiro para o resultado do parser. Cada posicao aponta para o trecho da entrada "parseada".
 * 	
 */
char** wordTokenizeDetails(const char* str)
{
	/* Inicializacao de variaveis */
	int count = 0; // Contador para a quantidade de tokes.
	int capacity = 10; // Tamanho do toke, utilizado para o alocamento de memoria, inicializado inicialmente em 10.
	char** result = malloc(capacity*sizeof(*result)); // Variavel de retorno.	
	const char* e=str; // Variavel auxiliar. 
	
	if (e) do 
	{
		const char* s=e;
		e=strpbrk(s,"'-|"); // Delimitadores do parser. Sempre que um desses caracteres for encontrado, sera criado um toke contendo o trecho de texto anterior a ele.
		
		if (count >= capacity)
			result = realloc(result, (capacity*=2)*sizeof(*result));
		
		result[count++] = e? strndup(s, e-s) : strdup(s);
	} while (e && *(++e));
	
	if (count >= capacity)
		result = realloc(result, (capacity+=1)*sizeof(*result));
	result[count++] = 0;
	
	return result;
}

/**
 *  	int wordPalindrome(char* buffer, char * argv, int* wordPrimeCount)
 * 
 * 	Descricao:
 * 		Esta funcao verifica a ocorrencia de palindromos em palavras e tambem calcula a quantidade de palindromos cuja soma do codigo ASCII seja primo.
 *
 * 	Parametros de entrada:
 * 		- buffer: Texto a ser analizado.
 * 		- argv: Nome do arquivo de entrada. Caso seja "wikipedia.txt", o calculo dos numeros primos sera realizado.
 * 		- wordPrimeCount: Ponteiro utilizado para retornar a quantidade de palindromos cuja soma do codigo ASCII seja primo.
 * 
 * 	Parametros de saida:
 * 		- (int): Quantidade de palindromos de palavras encotnrados.
 * 
 */
int wordPalindrome(char* buffer, char * argv, int* wordPrimeCount){
	
	/* Declaracao de variaveis */
	int i=0,k=0; // Variaveis de controle das estruturas de repeticao.  
	int palindromeCount=0; // Variavel que ira armazenar a quantidade de palindromos.
	int primeCount = 0; // Variavel que ira armazenar a quantidade de palindromos primos.
	
	/* Declaracao das variaveis utilizadas no parser e chamada a funcao */
	char** initialWordTokens = wordTokenize(buffer); // Variavel contendo os primeiros tokens de palavras.
	char** it; // Variavel auxiliar para se trabalhar com os tokens.
	
	for(it=initialWordTokens; it && *it; ++it){
		
		int initialWordLength = (int)strlen(*it); // Variavel que armazena o tamanho do token analisado.
		
		// Caso o token contenha mais que 2 caracteres, podemos verificar se ele e um palindromo.
		if(initialWordLength > 2){
			
			/* Declaracao de variaveis */ 
			char** finalWordTokens = wordTokenizeDetails(*it); // Variavel contendo os ultimos tokens de palavras. (Agora, sem "-" e "'")
			char** it2; // Variavel auxiliar para se trabalhar com os tokens.
			int finalWordLenght = 0; // Variavel que armazena o tamanho do token analisado.
			
			// Estrutura de repeticao que calcula o tamanho total da palavra, agora sem os caracteres "-" e "'"
			for(it2 = finalWordTokens; it2 && *it2; ++it2){				
				finalWordLenght+=(int)strlen(*it2);				
			}      			
			
			char* finalWord  = (char*) malloc(sizeof(char)*(finalWordLenght+1)); // Alocacao de memoria para a palavra final
			int firstIteration = 1; // Variavel de controle para a juncao dos tokens
			
			// Juncao dos tokens para que a palavra final nao contenha nenhum caracter especial.
			for(it2 = finalWordTokens; it2 && *it2; ++it2){
				if(firstIteration==1){
					strcpy(finalWord, *it2);
					firstIteration = 0;
				} else{
					strcat(finalWord, *it2);	  
				}				
				free(*it2);
			}			
			
			free(finalWordTokens);			
			
			// Caso o token final contenha mais que 2 caracteres, podemos verificar se ele e um palindromo.
			if(finalWordLenght > 2){
				
				int equalCount = 0; // Variavel de controle para a verificacao do palindromo.
				int asciiSum = 0; // Variavel que ira armazena a soma dos valores ASCII da palavra. 
				
				/* Passa dos os caracteres da palavra para letras minusculas */
				for(i=0; i<finalWordLenght; i++){
					finalWord[i] = tolower(finalWord[i]);
				}				
				
				/* Verificacao da ocorrencia de palindromo */
				for(i=0; i<finalWordLenght; i++){ // Caso os caracteres opostos nao sejam iguais (nao e palindromo)
					if(finalWord[i] != finalWord[finalWordLenght-i-1]){
						break;
					} else{ // Caso contrario (pode ser palindromo)
						equalCount++;
						
						if(equalCount == finalWordLenght){ // Caso a quantidade de caracteres opostos sejam iguais ao numero total de caracteres, a palavra e palindromo.
							
							palindromeCount++;
							
							/* Caso o arquivo de entrada seja o "wikipedia.txt", verifica se a soma ASCII e primo */
							if(!strcmp(argv, "wikipedia.txt")){
								
								/* Realiza a soma ASCII dos caracteres */
								for(k=0; k<finalWordLenght; k++){
									asciiSum += (int)finalWord[k];
								}
								
								/* Caso seja primo, o contador de palindromos primos e incrementado */
								if(crivoEratostenes(asciiSum)){
									primeCount++;																		
								}
								
								*wordPrimeCount = primeCount; // Retorno por ponteiro da quantidade de palindromos primos
							}
						}
					}
				}
				
			}
		}		
		
		free(*it); // Liberacao de memoria.		
	}
	
	free(initialWordTokens); // Liberacao de memoria.	
	
	return palindromeCount;
}

/**
 *  	int phrasePalindrome(char* buffer, char * argv, int* phrasePrimeCount)
 * 
 * 	Descricao:
 * 		Esta funcao verifica a ocorrencia de palindromos em frases e tambem calcula a quantidade de palindromos cuja soma do codigo ASCII seja primo.
 *
 * 	Parametros de entrada:
 * 		- buffer: Texto a ser analizado.
 * 		- argv: Nome do arquivo de entrada. Caso seja "wikipedia.txt", o calculo dos numeros primos sera realizado.
 * 		- wordPrimeCount: Ponteiro utilizado para retornar a quantidade de palindromos cuja soma do codigo ASCII seja primo.
 * 
 * 	Parametros de saida:
 * 		- (int): Quantidade de palindromos de frases encotnrados.
 * 	
 */
int phrasePalindrome(char* buffer, char * argv, int* phrasePrimeCount){
	
	/* Declaracao de variaveis */
	int i=0,k=0; // Variaveis de controle das estruturas de repeticao.  
	int palindromeCount=0; // Variavel que ira armazenar a quantidade de palindromos.
	int primeCount = 0; // Variavel que ira armazenar a quantidade de palindromos primos.
	
	/* Declaracao das variaveis utilizadas no parser e chamada a funcao */
	char** initialPhraseTokens = phraseTokenize(buffer);  // Variavel contendo os primeiros tokens de frases.
	char** it; // Variavel auxiliar para se trabalhar com os tokens.
	
	for(it=initialPhraseTokens; it && *it; ++it){
		
		/* Declaracao de variaveis */ 
		char** initialWordTokens = wordTokenize(*it); // Variavel que armazena o token da frase a ser analizada quebrada em palavras  
		char** it2; // Variavel auxiliar para se trabalhar com os tokens.
		int initialPhraseLength = 0; // Variavel que armazena o tamanho do token analisado.
		
		// Estrutura de repeticao que calcula o tamanho total da frase, agora sem os caracteres especiais
		for(it2=initialWordTokens; it2 && *it2; ++it2){			
			initialPhraseLength+=(int)strlen(*it2);     
		}
		
		/* Declaracao de variaveis */
		char* initialPhrase  = (char*) malloc(sizeof(char)*(initialPhraseLength+1)); // Declaracao e alocacao de memoria para a frase contendo somente as palavras sem caracteres especiais.
		int firstIteration = 1; // Variavel de controle para a juncao dos tokens
		
		/* Realiza a juncao dos tokens de palavras formando a frase inicial a ser analizada */
		for(it2=initialWordTokens; it2 && *it2; ++it2){			
			if(firstIteration==1){
				strcpy(initialPhrase, *it2);
				firstIteration = 0;
			} else{
				strcat(initialPhrase, *it2);	  
			}						
			free(*it2);
		}
		
		free(initialWordTokens);
		
		/* Declaracao de variaveis */ 
		char** finalWordTokens = wordTokenizeDetails(initialPhrase);  // Variavel que armazena o token da frase a ser analizada quebrada em palavras, agora, sem os caracteres "-" e "'"
		char** it3;  // Variavel auxiliar para se trabalhar com os tokens.
		int finalPhraseLength = 0; // Variavel que armazena o tamanho do token analisado.
		
		// Estrutura de repeticao que calcula o tamanho total da frase final
		for(it3=finalWordTokens; it3 && *it3; ++it3){
			finalPhraseLength+=(int)strlen(*it3);
		}
		
		/* Declaracao de variaveis */
		char* finalPhrase  = (char*) malloc(sizeof(char)*(finalPhraseLength+1)); // Declaracao e alocacao de memoria para a frase final.
		firstIteration = 1; // Variavel de controle para a juncao dos tokens
		
		/* Realiza a juncao dos tokens de palavras formando a frase final a ser analizada */
		for(it3=finalWordTokens; it3 && *it3; ++it3){
			
			if(firstIteration==1){
				strcpy(finalPhrase, *it3);
				firstIteration = 0;
			} else{
				strcat(finalPhrase, *it3);	
			}						
			free(*it3);
		}		
		
		free(finalWordTokens);		
		
		// Caso o token final contenha mais que 2 caracteres, podemos verificar se ele e um palindromo.
		if(finalPhraseLength > 2){
			
			int equalCount = 0; // Variavel de controle para a verificacao do palindromo.
			int asciiSum = 0; // Variavel que ira armazena a soma dos valores ASCII da frase. 
			
			/* Passa dos os caracteres da frase para letras minusculas */
			for(i=0; i<finalPhraseLength; i++){
				finalPhrase[i] = tolower(finalPhrase[i]);
			}
				
			/* Verificacao da ocorrencia de palindromo */
			for(i=0; i<finalPhraseLength; i++){ // Caso os caracteres opostos nao sejam iguais (nao e palindromo)
				if(finalPhrase[i] != finalPhrase[finalPhraseLength-i-1]){
					break;
				} else{ // Caso contrario (pode ser palindromo)
					equalCount++;
					
					if(equalCount == finalPhraseLength){ // Caso a quantidade de caracteres opostos sejam iguais ao numero total de caracteres, a palavra e palindromo.
						
						palindromeCount++;
						
						/* Caso o arquivo de entrada seja o "wikipedia.txt", verifica se a soma ASCII e primo */
						if(!strcmp(argv, "wikipedia.txt")){
							
							/* Realiza a soma dos codigo ASCII dos caracteres da frase */
							for(k=0; k<finalPhraseLength; k++){
								asciiSum += (int)finalPhrase[k];
							}
							
							/* Caso seja primo, o contador de palindromos primos e incrementado */
							if(crivoEratostenes(asciiSum)){
								primeCount++;								
							}
							
							*phrasePrimeCount = primeCount; // Retorno por ponteiro da quantidade de palindromos primos
						}
						
					}
				}
			}
			
		}				
		
		free(*it);	// Liberacao de memoria.				
	}
	
	free(initialPhraseTokens); // Liberacao de memoria.	
	
	return palindromeCount;
}
