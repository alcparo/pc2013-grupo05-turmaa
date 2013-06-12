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
 * 	Trabalho 3 - Geracao aleatoria de palavras
 *
 *	Este programa gera palavras aleatorias de ate 5 letras e verifica se ela
 * 	se encontra no arquivo texto de entrada. Palavras com mais de 5 letras sao
 * 	geradas a partir das palavras menores previamente encontradas.
 * 
 * 	Ele retorna os tempos de processamento de 3 etapas:
 * 		-Leitura do arquivo de entrada;
 * 		-Geracao das palavras pequenas;
 * 		-Combinação das palavras pequenas em palavras grandes.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strmap.h"
#include <mpi.h>
#include <omp.h>
#include <time.h>

#define TOTAL_ONE 26
#define TOTAL_TWO 196
#define TOTAL_THREE 847
#define TOTAL_FOUR 2590
#define TOTAL_FIVE 4441

char** wordTokenize(const char* str);
static void iter(const char *key, const char *value, const void *obj);
int generateIndex();
void addToBuffer(char* buffer, char* str);
void bufferDivision(char *buffer, char **bufferArray, int size);

	
int NUMTHREADS; // Variavel que ira armazenar a quantidade de threads utilizadas pelo openMP
int MPI_NUMTASKS, MPI_RANK, tag=1;

/**
 *  	int main(int argc, char * argv[])
 * 
 *	Descricao:
 * 		Função principal. Nela esta codificada a nossa "topologia" para o problema:
 * 			- No mestre: Responsável pela leitura e processamento do arquivo de entrada,
 * 			bem como enviar e receber informações dos outros nós.
 * 			- Nos intermediarios: Geram as palavras aleatoriamente.
 * 			- No final: Gera as combinações possiveis de palavras grandes utilizando as palavras
 * 			pequenas encontradas.
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
	
	MPI_Status status;	
	
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_NUMTASKS);	
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
	
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
	
	if(MPI_NUMTASKS <= 2){
		printf("Erro: E necessario a execucao com pelo menos 3 nos.\n");
		exit(1);
	}		
	
	if (MPI_RANK == 0) {
		double timeStart1 = omp_get_wtime( ); // Armazena o tempo de inicio do programa, em segundos.	
	
		int i; // Variavel de controle para as estrutudas de repeticao.
		
		//ENTRADA
		FILE * pFile; // Variavel para a abertura do arquivo de entrada.
		long lSize; // Variavel que armazena o tamanho do arquivo de entrada.
		char * buffer; // Variavel que ira armazenar o arquivo de entrada na memoria.
		size_t result; // Variavel que ira armazenar o retorno da funcao de leitura do arquivo de entrada. 
		
		//printf("Abre arquivo\n");
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
		
		fclose(pFile);
		//printf("Fecha arquivo\n");
		
		
		
		StrMap *smSmall, *smBig;
		char buf[255];
		//HASH TABLE : evitar repeticao 
		smSmall = sm_new(lSize);
		smBig = sm_new(lSize);
		if (smSmall == NULL || smBig == NULL ) {
			exit(1);
		}
		
		char * bufferBig = (char*)malloc(sizeof(char)*lSize);
		char * bufferSmall = (char*)malloc(sizeof(char)*lSize);
		
		int wordBigCount=0, wordSmallCount=0;
		int wordLength=0;	
			
		char** words = wordTokenize(buffer);
		char** it;
		
		//printf("Inicia filtragem\n");
		char nullChar[1]={'\0'};
	
		for(it=words; it && *it; ++it){

			wordLength=(int)strlen(*it);	
			
			if(wordLength>5 && !sm_get(smBig, *it, buf, sizeof(buf))){
				sm_put(smBig, *it, *it);
				wordBigCount++;
				addToBuffer(bufferBig, *it);
				
			} else	if(wordLength<=5 && !sm_get(smSmall, *it, buf, sizeof(buf)) && strcmp(*it, nullChar)!=0){
				sm_put(smSmall, *it, *it);
				wordSmallCount++;
				addToBuffer(bufferSmall, *it);
			}			
			
		}
		
		//printf("\n\n\n%s\n\n\n", bufferSmall);
		printf("Palavras grandes: %d\tPalavras pequenas: %d\tTotal: %d\n", wordBigCount, wordSmallCount, wordSmallCount+wordBigCount);
		sm_delete(smBig);
		sm_delete(smSmall);
		
		//printf("Acaba filtragem\n");
		
		bufferBig = (char*)realloc(bufferBig, sizeof(char)*strlen(bufferBig));
		bufferSmall = (char*)realloc(bufferSmall, sizeof(char)*strlen(bufferSmall));
	
		
		double timeEnd1 = omp_get_wtime( );		
		printf("Tempo de leitura e processamento: %lf\n",timeEnd1-timeStart1);
		
		
		// ENVIO BUFFER GRANDE P ULTIMO NO
		int lengthBufferBig = (int)strlen(bufferBig);
		MPI_Send(&lengthBufferBig, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
		MPI_Send(bufferBig, lengthBufferBig, MPI_CHAR, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);		
		MPI_Send(&wordBigCount, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
		

		
		int ARRAYSIZE = MPI_NUMTASKS-2;
		char* bufferArray[ARRAYSIZE];		
		bufferDivision(bufferSmall, bufferArray, ARRAYSIZE);
	
		
		int lengthBufferArray;
		//ENVIO DAS PALAVRAS PEQUENAS PARA SEUS RESPECTIVOS NOS
		#pragma omp for
		for(i=1; i<MPI_NUMTASKS-1; i++){
			lengthBufferArray = (int)strlen(bufferArray[i-1])+1; // +1 pra enviar o \0 e nao pegar lixo
			MPI_Send(&lengthBufferArray, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(bufferArray[i-1], lengthBufferArray, MPI_CHAR, i, tag, MPI_COMM_WORLD);		
		}	
	
		//RECEBE O TOTAL DE PALAVRAS GRANDES ENCONTRADAS
		int bigWordsFound, smallWordsFound;
		double timeMeans[10];
		MPI_Recv(&smallWordsFound, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD, &status);				
		MPI_Recv(timeMeans, 10, MPI_DOUBLE, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&bigWordsFound, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD, &status);	
		
		printf("\nPalavras grandes encontradas: %d\tPalavras pequenas encontradas: %d\n", bigWordsFound ,smallWordsFound);
		printf("Tempos de execução medio por %% de palavras pequenas encontradas:\n");
		for(i=0; i<10; i++) printf("%d%%\t%lf\n", (i+1)*10, timeMeans[i]);
		printf("\n");
		
	
	} else if(MPI_RANK!=(MPI_NUMTASKS-1)){
		double startTime2;
		double endTimeAux2;
	
		
		char buf[256];
		srand (time(NULL)/MPI_RANK);

		int index;
		int i, r;
		
		
		
		double times[10];

		
		//RECEBE OS BUFFERS DE PALAVRAS PEQUENAS
		int lengthBufferSmall;
		MPI_Recv(&lengthBufferSmall, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);		
		char* bufferSmall = (char*)malloc(sizeof(char)*lengthBufferSmall);
		MPI_Recv(bufferSmall, lengthBufferSmall, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
	
		char** wordsAux = wordTokenize(bufferSmall);
		char** itAux;
		StrMap *smSmall;
		
		
		int wordCount=0;
		
		for(itAux=wordsAux; itAux && *itAux; ++itAux){
			wordCount++;

		}

		
		smSmall = sm_new(wordCount);
		for(itAux=wordsAux; itAux && *itAux; ++itAux){
			
			sm_put(smSmall, *itAux, *itAux);
		}

		
		char* bufferFoundWords = (char*)malloc(sizeof(char)*lengthBufferSmall);

		
		int percent, percentFound=1;
		
		
		
		int foundWords = 0;
		startTime2 = omp_get_wtime( );
		while(foundWords<(wordCount)){			
				
			
				index = generateIndex();

				char randomWord[index+1];
				
				#pragma omp for
				for(i=0; i<index; i++){
					r = 97 + (rand() % 26); //rand() % 26 entre 0 e 25 - 97 a 122
					randomWord[i] = (char)r;
				}	
				
				randomWord[index]='\0';
				
				if(sm_get(smSmall, randomWord, buf, sizeof(buf)) && strcmp(buf,"-1")){
					
					
						foundWords++;
						//if(foundWords>=(wordCount*0.95)) printf("%d\n", foundWords);
						
						sm_put(smSmall, randomWord, "-1");
						addToBuffer(bufferFoundWords, randomWord);
					
						for(percent=percentFound; percent<=10; percent++){
							
							if(foundWords == (int)(wordCount*((float)percent/10))){
								
								
								endTimeAux2 = omp_get_wtime( );
								
								times[percent-1] = (endTimeAux2-startTime2);
								percentFound++;
								
							}
						}
						
				
				}
			
			
		
		}
		
		int sizeOfBufferFoundWords = (int)strlen(bufferFoundWords)+1;

		MPI_Send(&foundWords, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);		
		MPI_Send(&sizeOfBufferFoundWords, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
		MPI_Send(bufferFoundWords, sizeOfBufferFoundWords, MPI_CHAR, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
		MPI_Send(times, 10, MPI_DOUBLE, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
			
	} else {
		int i, k;
		double timeMeans[10];
		
		for(i=0; i<10; i++) timeMeans[i]=0;
		
		char buf[256];
		
		//BUFFER PALAVRAS GRANDES
		
		int lengthBufferBig;
		MPI_Recv(&lengthBufferBig, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);		
		char* bufferBig = (char*)malloc(sizeof(char)*lengthBufferBig);
		MPI_Recv(bufferBig, lengthBufferBig, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
		
		int wordBigCount;
		MPI_Recv(&wordBigCount, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		
		
		//bufferBig = (char*)realloc(bufferBig, sizeof(char)*strlen(bufferBig));
		
		char** wordsAux = wordTokenize(bufferBig);
		char** itAux;
		
		StrMap *smBig;
		smBig = sm_new(wordBigCount);
		
		for(itAux=wordsAux; itAux && *itAux; ++itAux){

			sm_put(smBig, *itAux, *itAux);
		}
		
		
		int ARRAYSIZE = NUMTHREADS;
		char* bufferArray[ARRAYSIZE];		
		bufferDivision(bufferBig, bufferArray, ARRAYSIZE);
		
		
		
		//for(i=0; i<ARRAYSIZE; i++)printf("PROCESS: %d - %d\n", i, (int)strlen(bufferArray[i]));
		
		
		//RECEBE O BUFFER DE PALAVRAS ACHADAS NOS OUTROS NOS	
		int lengthBufferSmall = 0;
		int lengthBufferSmallAux;
		int smallWordsFoundAux = 0;
		int smallWordsFound = 0;
		
		char* bufferAux[MPI_NUMTASKS-2];
		
		
		double *times[MPI_NUMTASKS-2];
		for(i=0; i<MPI_NUMTASKS-2; i++)
			times[i] = (double*)malloc(sizeof(double)*10);
		
		//double times[MPI_NUMTASKS-2][10];
		
		#pragma omp for
		for(i=1; i<MPI_NUMTASKS-1; i++){
			
			MPI_Recv(&smallWordsFoundAux, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			smallWordsFound+=smallWordsFoundAux;
			
			MPI_Recv(&lengthBufferSmallAux, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			
			
			bufferAux[i-1] = (char*)malloc(sizeof(char)*lengthBufferSmallAux);
			MPI_Recv(bufferAux[i-1], lengthBufferSmallAux, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
			lengthBufferSmall+=lengthBufferSmallAux;
			
			MPI_Recv(times[i-1], 10, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
			
		}
		
		
		double timeAux;					
	
		for(i=0;i<10;i++){
			timeAux=0;
			for(k=0;k<MPI_NUMTASKS-2;k++){
				timeAux+=times[k][i];
			}
			timeMeans[i] = timeAux/(MPI_NUMTASKS-2);
		}
		
	
		
		MPI_Send(&smallWordsFound, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);	
		MPI_Send(timeMeans, 10, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);	
		
		
		char* bufferSmallFound = (char*)malloc(sizeof(char)*lengthBufferSmall);

		
		for(i=1; i<MPI_NUMTASKS-1; i++){
			memcpy(bufferSmallFound+strlen(bufferSmallFound), bufferAux[i-1], strlen(bufferAux[i-1]));
		}
				
		bufferSmallFound = (char*)realloc(bufferSmallFound, sizeof(char)*strlen(bufferSmallFound));
		
	
		
		int bigWordsFoundAux, bigWordsFound=0, countAux;

		
		//ADICAO PALAVRAS ACHADAS NA TABELA HASH
		char** wordsAux2 = wordTokenize(bufferSmallFound);
		char** itAux2;

		StrMap *smSmall;
		smSmall = sm_new(smallWordsFound);
	
		for(itAux2=wordsAux2; itAux2 && *itAux2; ++itAux2){

			sm_put(smSmall, *itAux2, *itAux2);
		}
		
		
		double startTime3 = omp_get_wtime( );
		double endTimeAux3;
		//printf("Inicio combinacoes\n");
		#pragma omp parallel private(bigWordsFoundAux, countAux) shared(smSmall) reduction(+: bigWordsFound)
		{	
			
			
			bigWordsFoundAux=0;
			char** words = wordTokenize(bufferArray[omp_get_thread_num()]);
			char** it;
				
			int k, l;
			int lengthAux;
			int lengthAuxDivided;
			
			
			char wordAux[6];
			
				for(it=words; it && *it; ++it){
					
					for(k=5; k>0; k--){
						wordAux[0]=wordAux[1]=wordAux[2]=wordAux[3]=wordAux[4]=wordAux[5]='\0';

						lengthAux = (int)strlen(*it);
						if(!(lengthAux%k)){
							lengthAuxDivided = (int)lengthAux/k;
						} else{
							lengthAuxDivided = ((int)lengthAux/k)+1;
						}
						
						countAux = lengthAuxDivided;
						
						
						for(l=0; l<lengthAuxDivided; l++){
						
							strncpy(wordAux, *it+(l*k), k);					
							
							if(sm_get(smSmall, wordAux, buf, sizeof(buf))){
									countAux--;
							} else{
								break;
							}
						}
						
						if(countAux == 0){

							bigWordsFoundAux++;
							#pragma omp critical
							{
							sm_put(smBig, *it, "-1");
							}
							break;
						}
				
					}
				}
				
			
			bigWordsFound+=bigWordsFoundAux;
		}
		endTimeAux3 = omp_get_wtime( );
		printf("Tempo de combinacao: %lf\n",endTimeAux3-startTime3);
		
		MPI_Send(&bigWordsFound, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
				
		
	}
	
	
	MPI_Finalize();

	return 0;
}


/**
 *  	void addToBuffer(char* buffer, char* str)
 * 
 *	Descricao:
 * 		Adiciona a string str no fim do buffer. Ao adicionar insere um '\n' para
 *		que cada string adicionada fique numa linha.	
 * 
 * 	Parametros de entrada:
 * 		- char* buffer: Buffer a ser utilizado.
 * 		- char* str: String a ser inserida.
 * 
 * 	Parametros de saida:
 * 		- 
 * 	
 */
void addToBuffer(char* buffer, char* str){
	size_t bufferLenght, strLength;
	char n[1]={'\n'};
	char end[1]={'\0'};
	
	bufferLenght=strlen(buffer);
	strLength=strlen(str);
	memcpy(buffer+bufferLenght, str, strLength);
	memcpy(buffer+bufferLenght+strLength, n, sizeof(char));
	memcpy(buffer+bufferLenght+strLength+1, end, sizeof(char));
}

/**
 *  	generateIndex()
 * 
 *	Descricao:
 * 		Gera aleatoriamente um tamanho para a palavra aleatoria. Para cada tamanho
 * 		e atribuido um peso a fim de gerar tamanhos maiores de uma forma mais frequente.
 * 		SOlucao discutida com outros grupos.
 * 
 * 	Parametros de entrada:
 * 		- 
 * 
 * 	Parametros de saida:
 * 		- int: Tamanho da palavra a ser gerada, de 1 a 5.
 * 	
 */
int generateIndex(){
	
	unsigned length;
	length = rand()%TOTAL_FIVE;
	if (length > TOTAL_FOUR)
		length = 5;	
	else if (length > TOTAL_THREE)
		length = 4;
	else if (length > TOTAL_TWO)
		length = 3;
	else if (length > TOTAL_ONE)
		length = 2;
	else 
		length = 1;
	
	return length;	
}

/**
 *  	void bufferDivision(char *buffer, char **bufferArray, int size)
 * 
 *	Descricao:
 * 		Realiza a divisao do buffer em grupos de tamanho size. O delimitador para essa divisao
 * 		e o '\n'.
 * 
 * 	Parametros de entrada:
 * 		- char* buffer: Buffer a ser dividido.
 * 		- char **bufferArray: Array que ira armazenar as divisoes dos buffers.
 * 		- int size: A quantidade de grupos que o buffer sera dividido.
 * 
 * 	Parametros de saida:
 * 		- 
 * 	
 */
void bufferDivision(char *buffer, char **bufferArray, int size){
	int ARRAYSIZE = size;
	int bufferLength = (int)strlen(buffer);	
	int i;
	/* Declaracao de variaveis */
	int bufferBegin[ARRAYSIZE], bufferEnd[ARRAYSIZE]; // Variaveis que irao armazenar o inicio e fim de cada particao do texto de entrada.
	int bufferSize = (bufferLength / ARRAYSIZE); // Variavel que ira armazena o tamanho inicial de cada particao do arquivo de entrada.
	
	int divided = 0; // Variavel de controle para verificar se o arquivo de entrada terminou de ser particionado.
	
	/* Estrutura de repeticao que realiza o calculo das posicoes de inicio e fim iniciais das particoes do arquivo de entrada */
	bufferBegin[0] = 0; bufferEnd[0] = bufferSize;
	for(i=1; i<ARRAYSIZE; i++){
		bufferBegin[i] = bufferEnd[i-1]+1;
		bufferEnd[i] = bufferBegin[i]+bufferSize;
		if(i==ARRAYSIZE-1)
			bufferEnd[i]=bufferLength;
	}		
	
	/* Estrutura de repeticao que realiza a particao do arquivo de entrada em n particoes, onde n = ARRAYSIZE */
	for(i=0; i<ARRAYSIZE-1; i++){
		divided = 0;

		while(!divided){
			if(buffer[bufferEnd[i]] == '\n'){ // Verifica se o caracter verificado e um delimitador de frase
				bufferArray[i] = strndup(buffer+bufferBegin[i], bufferEnd[i]-bufferBegin[i]+1); // +1  pra pegar o ponto em si
				divided = 1;
			} else{ // Caso nao seja um delimitador de frase, atualiza as posicoes das particoes.
				bufferEnd[i]++;
				bufferBegin[i+1]++;				
			}
		}		
	}
	
	/* Atribuicao da ultima particao */
	bufferArray[ARRAYSIZE-1] = strndup(buffer+bufferBegin[ARRAYSIZE-1], bufferEnd[ARRAYSIZE-1]-bufferBegin[ARRAYSIZE-1]+1);

}

/**
 *  	char** wordTokenize(const char* str)
 * 
 Descricao:
 * 		Funcao que realiza o parser do arquivo de entrada em palavras.
		Este codigo, mais otimizado do que simplesmente utilizar "strtok()", foi encontrado no link a seguir e, 
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
		e=strpbrk(s,"\n"); // Delimitadores do parser. Sempre que um desses caracteres for encontrado, sera criado um toke contendo o trecho de texto anterior a ele.
		
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
 *  	static void iter(const char *key, const char *value, const void *obj)
 * 
 *	Descricao:
 * 		Funcao de iteracao utilizada para enumerar todos os itens da hash table.
 *		Funcao utilizada pelo grupo apenas para begug.	
 * 	
 * 	Parametros de entrada:
 * 		- const char *key: Chave da hash table.
 * 		- const char * value: Valor referente a chave.
 * 		- void* obj: Se necessario, objeto a ser passado para funcao e posteriormente retornado.
 * 
 * 	Parametros de saida:
 * 		- 
 * 	
 */
static void iter(const char *key, const char *value, const void *obj)
{
    printf("key: %s value: \t\t\t%s\n", key, value);
}