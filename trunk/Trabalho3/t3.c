#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strmap.h"
#include <mpi.h>
#include <omp.h>
#include <time.h>

#define TOTAL_ONE 24
#define TOTAL_TWO 196
#define TOTAL_THREE 847
#define TOTAL_FOUR 2590
#define TOTAL_FIVE 4441

char** wordTokenize(const char* str);
char* generateWord(int length);
static void iter(const char *key, const char *value, const void *obj);
int generateIndex();
void addToBuffer(char* buffer, char* str);
void bufferDivision(char *buffer, char **bufferArray, int size);

	
int NUMTHREADS; // Variavel que ira armazenar a quantidade de threads utilizadas pelo openMP
int MPI_NUMTASKS, MPI_RANK, tag=1;



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
		
		
		
		
		
		//DIVISAO DO BUFFER DE PALAVRAS PEQUENAS EM N PEDACOS (N = NUMERO DE NOS INTERMEDIARIOS)
		//int lengthBufferSmall = (int)strlen(bufferSmall);
		
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
		MPI_Recv(&smallWordsFound, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD, &status);		
		MPI_Recv(&bigWordsFound, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD, &status);		
		
		
		//printf("\nTOTAL SMALL: %d\n\n", smallWordsFound);
		//printf("\nTOTAL BIG: %d\n\n", bigWordsFound);
		
		//double timeEnd = omp_get_wtime( );	
		//printf("Execution time: %lf\n",timeEnd-timeStart);
		//printf("FIM\n");
		
		//sm_enum(smSmall, iter, NULL);
		
	
	
	} else if(MPI_RANK!=(MPI_NUMTASKS-1)){
		double startTime2 = omp_get_wtime( );
		double endTimeAux2;
	
		
		
		char buf[256];
		srand (time(NULL)/MPI_RANK);

		int index;
		int i, r;
		
		
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
		
		//printf("word count\t%d\n", wordCount);
		
		smSmall = sm_new(wordCount);
		for(itAux=wordsAux; itAux && *itAux; ++itAux){
			
			sm_put(smSmall, *itAux, *itAux);
		}
		
		//sm_enum(smSmall, iter, NULL);
		
		char* bufferFoundWords = (char*)malloc(sizeof(char)*lengthBufferSmall);
		//char end[1]={'\0'};
		//strcpy(bufferAchadas, end);
		int percent, percentFound=1;
		
		
		
		int foundWords = 0;
		
		while(foundWords<(wordCount)){			
				
			
				index = generateIndex();
				
				
				//char* word = (char*) malloc(sizeof(char)*(length+1));
				char randomWord[index+1];
				
				#pragma omp for
				for(i=0; i<index; i++){
					r = 97 + (rand() % 26); //rand() % 26 entre 0 e 25 - 97 a 122
					randomWord[i] = (char)r;
				}	
				
				randomWord[index]='\0';
				
				if(sm_get(smSmall, randomWord, buf, sizeof(buf)) && strcmp(buf,"-1")){
					//#pragma omp critical
					//{
						foundWords++;
						//if(foundWords>=(wordCount*0.95)) printf("%d\n", foundWords);
						
						sm_put(smSmall, randomWord, "-1");
						addToBuffer(bufferFoundWords, randomWord);
					//}
						for(percent=percentFound; percent<=10; percent++){
							
							if(foundWords == (int)(wordCount*((float)percent/10))){
									percentFound++;
								endTimeAux2 = omp_get_wtime( );
								printf("Node %d\t%d %%\t%lf s\n", MPI_RANK, (percent*10), endTimeAux2-startTime2);
							}
						}
						
				
				}
			
			
		
		}
		
		//printf("RANK %d\t PROCESSO %d\t%d\n", MPI_RANK, omp_get_thread_num(), foundWords);
					

		//bufferAchadas = (char*)realloc(bufferAchadas, sizeof(char)*strlen(bufferAchadas));
		int sizeOfBufferFoundWords = (int)strlen(bufferFoundWords)+1;

		MPI_Send(&foundWords, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);		
		MPI_Send(&sizeOfBufferFoundWords, 1, MPI_INT, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
		MPI_Send(bufferFoundWords, sizeOfBufferFoundWords, MPI_CHAR, MPI_NUMTASKS-1, tag, MPI_COMM_WORLD);
		
		
		//endTime2 = omp_get_wtime( );
		
	} else {
		int i;
	
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
		
		#pragma omp for
		for(i=1; i<MPI_NUMTASKS-1; i++){
			
			MPI_Recv(&smallWordsFoundAux, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			smallWordsFound+=smallWordsFoundAux;
			
			MPI_Recv(&lengthBufferSmallAux, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			
			
			bufferAux[i-1] = (char*)malloc(sizeof(char)*lengthBufferSmallAux);
			MPI_Recv(bufferAux[i-1], lengthBufferSmallAux, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
			lengthBufferSmall+=lengthBufferSmallAux;
			//printf("%s\n\n", bufferAux[i-1]);
		}
		//printf("PALAVRAS ACHADAS: %d\n\n", smallWordsFound);
		
		
		MPI_Send(&smallWordsFound, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);	
		
		
		
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
			
		//printf("CONSTRUIU\n");
	
		double startTime3 = omp_get_wtime( );
		double endTimeAux3;
		//printf("Inicio combinacoes\n");
		#pragma omp parallel private(bigWordsFoundAux, countAux) shared(smSmall) reduction(+: bigWordsFound)
		{	
			
			//printf("THREAD: \t%d\n\n", omp_get_thread_num());
			
			bigWordsFoundAux=0;
			char** words = wordTokenize(bufferArray[omp_get_thread_num()]);
			char** it;
			//printf("%s\n", bufferBig);
			
			int k, l;
			int lengthAux;
			int lengthAuxDivided;
			
			//char end[1]={'\0'};
			char wordAux[6];
			
			//wordAux[0]=wordAux[1]=wordAux[2]=wordAux[3]=wordAux[4]=wordAux[5]='\0';
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
							//char* wordAux=(char*)malloc(sizeof(char)*6);
							strncpy(wordAux, *it+(l*k), k);						
							//strcat(wordAux, end);
							//wordAux[5]='\0';
							//printf("%s\t", wordAux);
							
							if(sm_get(smSmall, wordAux, buf, sizeof(buf))){
								//printf("%s\t", wordAux);
									countAux--;
							} else{
								break;
							}
						}
						
						if(countAux == 0){
							//printf("ACHOU!: %s\n", *it);
							bigWordsFoundAux++;
							#pragma omp critical
							{
							sm_put(smBig, *it, "-1");
							}
							break;
						}
						//printf("\n");
					}
				}
				
			
			bigWordsFound+=bigWordsFoundAux;
		}
		endTimeAux3 = omp_get_wtime( );
		printf("Tempo de combinacao: %lf\n",endTimeAux3-startTime3);
		
		//printf("Fim combinacoes\n");
			

		//printf("STOP\n");
		//printf("RANK %d TERMINOU\tFOUND BIG: %d\n", MPI_RANK, bigWordsFound);
		MPI_Send(&bigWordsFound, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
				
		
	}
	
	//printf("RANK %d TERMINOU\n", MPI_RANK);
		

	MPI_Finalize();

	return 0;
}



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

int generateIndex(){
	//int r = 1+(rand()%5);	
	//return r;
	
	
	
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
	
	//printf("%d\n", length);
	return length;
	
	
}

char* generateWord(int length){
	
	int i;
	int r;
	char* word = (char*) malloc(sizeof(char)*(length+1));
	
	for(i=0; i<length; i++){
		r = 97 + (rand() % 26); //rand() % 26 entre 0 e 25 - 97 a 122
		word[i] = (char)r;
	}	
	
	word[length]='\0';
	
	return word;
}



void bufferDivision(char *buffer, char **bufferArray, int size){
	int ARRAYSIZE = size;
	int bufferLength = (int)strlen(buffer);	
	int i;
	/* Declaracao de variaveis */
	int bufferBegin[ARRAYSIZE], bufferEnd[ARRAYSIZE]; // Variaveis que irao armazenar o inicio e fim de cada particao do texto de entrada.
	int bufferSize = (bufferLength / ARRAYSIZE); // Variavel que ira armazena o tamanho inicial de cada particao do arquivo de entrada.
	//bufferArray[ARRAYSIZE]; // Array que ira armazenar cada particao do arquivo de entrada.
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

static void iter(const char *key, const char *value, const void *obj)
{
    printf("key: %s value: \t\t\t%s\n", key, value);
}