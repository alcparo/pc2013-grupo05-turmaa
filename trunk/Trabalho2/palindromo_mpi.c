#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include <time.h>

int phrasePalindrome(char* buffer, char * argv, int* phrasePrimeCount);
int wordPalindrome(char* buffer, char * argv, int* wordPrimeCount);
char** wordTokenize(const char* str);
char** wordTokenizeDetails(const char* str);
char** phraseTokenize(const char* str);
int crivoEratostenes(int num);

int main(int argc, char * argv[]){
	
	int i;
	
	FILE * pFile;
	long lSize;
	char * buffer;
	size_t result;
	
	if((argc < 2) || (pFile = fopen(argv[1],"r")) == NULL){
		fputs ("File error",stderr); exit (1);
	}
	
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
	
	// allocate memory to contain the whole file:
	buffer = (char*) malloc (sizeof(char)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
	
	// copy the file into the buffer:
	result = fread (buffer,1,lSize,pFile);
	rewind (pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	
	int wordPalindromeCount, phrasePalindromeCount;
	int* wordPrimeCount = (int*)malloc(sizeof(int));	*wordPrimeCount = 0;
	int* phrasePrimeCount = (int*)malloc(sizeof(int));	*phrasePrimeCount = 0;
	
	//rank = id do processo, source = remetente, tag = id da mensagem
	int rank, source, tag=1;
	//buffer da rede
	char *bufferenvio, *bufferrec;
	char *buffer1, *buffer2, *buffer3, *buffer4;
	//divisao = divisao do arquivo em 4 partes(0 < 1/4 < 2/4 < 3/4 < 1), posicao = posicao onde vai dividir	
	int divisao, posicao, inicio=0; 
	//nome da maquina
	int nameSize;
	//armazena informacoes da mensagem
	MPI_Status status;
	//inicializa a sessao MPI
	MPI_Init(&argc, &argv);
	//obtem o id do processo
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	char computerName[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(computerName, &nameSize);
	fprintf(stderr, "%s - Processo %d\n", computerName, rank);
	int total;
	double start_t, end_t;
	//rank = 0 -> processo master

	start_t = MPI_Wtime();
	if(rank==0){
		int s1, wordPalindromeCountTot=0, s2, phrasePalindromeCountTot=0, s3, s4, wordPrimeCountTot=0, phrasePrimeCountTot=0;
		for(i=1;i<5;i++){
			divisao=(i*lSize)/4;				
			posicao = partition(divisao, buffer, lSize);

			//fim de arquivo
			if(!posicao){
				bufferenvio = (char*) malloc (sizeof(char)*(lSize-inicio));
				total = lSize-inicio;	
				fread(bufferenvio, 1, (lSize-inicio), pFile);
			}
			//c.c.
			else{
				bufferenvio = (char*) malloc (sizeof(char)*(posicao-inicio));		
				total = posicao-inicio;	
				fread(bufferenvio, 1, (posicao-inicio), pFile);
				//atualiza o ponteiro do arquivo
				fseek (pFile , posicao+1 , 0);
				//atualiza o novo inicio		
				inicio = posicao+1;
			}
			MPI_Send(&wordPrimeCount, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD);
			MPI_Send(&phrasePrimeCount, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD);
			MPI_Send(&total, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(bufferenvio, total, MPI_CHAR, i, tag, MPI_COMM_WORLD);
		}
		for(i=1;i<5;i++){
			MPI_Recv(&s1, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			wordPalindromeCountTot+=s1;
			MPI_Recv(&s2, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			phrasePalindromeCountTot+=s2;
			if(!strcmp(argv[1], "wikipedia.txt")){
				MPI_Recv(&s3, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
				wordPrimeCountTot+=s3;
				MPI_Recv(&s4, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
				phrasePrimeCountTot+=s4;
			}
		}
		end_t = MPI_Wtime();
		printf("Tempo total: %f\n", end_t-start_t);
		if(!strcmp(argv[1], "shakespe.txt")){		
			printf("%s\nWord palindromes: %d\nPhrase palindromes: %d\n", argv[1], wordPalindromeCountTot, phrasePalindromeCountTot);
		} else if(!strcmp(argv[1], "wikipedia.txt")){				
			printf("%s\nWord palindromes: %d\t\tWord prime palindromes: %d\nPhrase palindromes: %d\t\tPhrase prime palindromes: %d\n",argv[1], wordPalindromeCountTot, wordPrimeCountTot, phrasePalindromeCountTot, phrasePrimeCountTot);
		}
	}
	//outros 4 processos irão receber e processar cada parte da string	
	else{
		source=0;
		//recebe os valores
		MPI_Recv(&wordPrimeCount, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&phrasePrimeCount, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&total, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
		bufferrec = malloc(total*sizeof(char));
		MPI_Recv(bufferrec, total, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);

		//processa
		wordPalindromeCount = wordPalindrome(bufferrec, argv[1], wordPrimeCount);
		phrasePalindromeCount = phrasePalindrome(bufferrec, argv[1], phrasePrimeCount);
		//printf("recebendo do processo %d", source);
		MPI_Send(&wordPalindromeCount, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		MPI_Send(&phrasePalindromeCount, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		
		if(!strcmp(argv[1], "wikipedia.txt")){		
			MPI_Send(wordPrimeCount, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
			MPI_Send(phrasePrimeCount, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		}
	}
	
	MPI_Finalize();
	free(buffer);	
	free(wordPrimeCount);	
	free(phrasePrimeCount);	
	return 0;
	
}


int partition(int novoinicio, char *buffer, int lSize){
	int i;
	for(i=novoinicio;i<lSize;i++){
		if(buffer[i]=='!' || buffer[i]=='.' || buffer[i]=='?'){
			return i;
		}
	}
	return 0;
}


int crivoEratostenes(int num){
	int primo = 0;
	int i, j;
	
	int* vetor = (int*) malloc(sizeof(int)*(num+1));
	
	for(i=2; i<=num; i++)
		vetor[i] = i;
	
	for(i=2; i<=num; i++){
		if(vetor[i] == i){ //eh primo
			
			for(j=i*i; j<=num; j+=i){
				vetor[j] = 0;
			}
			
			if(i == num){
				primo=1;
				break;
			}
	}
	
}

free(vetor);

return primo;
}

//http://stackoverflow.com/questions/8106765/using-strtok-in-c
char** phraseTokenize(const char* str)
{
	int count = 0;
	int capacity = 10;
	char** result = malloc(capacity*sizeof(*result));
	
	const char* e=str;
	
	if (e) do 
	{
		const char* s=e;
		e=strpbrk(s,".!?");
		
		if (count >= capacity)
			result = realloc(result, (capacity*=2)*sizeof(*result));
		
		result[count++] = e? strndup(s, e-s) : strdup(s);
	} while (e && *(++e));
	
	if (count >= capacity)
		result = realloc(result, (capacity+=1)*sizeof(*result));
	result[count++] = 0;
	
	return result;
}

char** wordTokenize(const char* str)
{
	int count = 0;
	int capacity = 10;
	char** result = malloc(capacity*sizeof(*result));
	
	const char* e=str;
	
	if (e) do 
	{
		const char* s=e;
		e=strpbrk(s," \r\n\t\".,;:(){}[]0123456789!~^><+=\\&*/_#");
		
		if (count >= capacity)
			result = realloc(result, (capacity*=2)*sizeof(*result));
		
		result[count++] = e? strndup(s, e-s) : strdup(s);
	} while (e && *(++e));
	
	if (count >= capacity)
		result = realloc(result, (capacity+=1)*sizeof(*result));
	result[count++] = 0;
	
	return result;
}

char** wordTokenizeDetails(const char* str)
{
	int count = 0;
	int capacity = 10;
	char** result = malloc(capacity*sizeof(*result));
	
	const char* e=str;
	
	if (e) do 
	{
		const char* s=e;
		e=strpbrk(s,"'-|");
		
		if (count >= capacity)
			result = realloc(result, (capacity*=2)*sizeof(*result));
		
		result[count++] = e? strndup(s, e-s) : strdup(s);
	} while (e && *(++e));
	
	if (count >= capacity)
		result = realloc(result, (capacity+=1)*sizeof(*result));
	result[count++] = 0;
	
	return result;
}

int wordPalindrome(char* buffer, char * argv, int* wordPrimeCount){
	int i=0,k=0; //laco  
	int palindromeCount=0;
	int primeCount = 0;
	
	
	char** initialWordTokens = wordTokenize(buffer);  
	char** it;	
	
	for(it=initialWordTokens; it && *it; ++it){
		
		int initialWordLength = (int)strlen(*it);
		
		
		if(initialWordLength > 2){
			
			char** finalWordTokens = wordTokenizeDetails(*it);
			char** it2;
			int finalWordLenght = 0;
			
			for(it2 = finalWordTokens; it2 && *it2; ++it2){
				
				finalWordLenght+=(int)strlen(*it2);
				
			}      			
			
			char* finalWord  = (char*) malloc(sizeof(char)*(finalWordLenght+1));
			int firstIteration = 1;
			
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
			
			
			if(finalWordLenght > 2){
				
				int equalCount = 0;
				int asciiSum = 0;
				
				for(i=0; i<finalWordLenght; i++){
					finalWord[i] = tolower(finalWord[i]);
				}				
				
				for(i=0; i<finalWordLenght; i++){
					if(finalWord[i] != finalWord[finalWordLenght-i-1]){
						break;
					} else{
						equalCount++;
						
						if(equalCount == finalWordLenght){
							
							palindromeCount++;
							//  printf("%s : ", finalWord);
							if(!strcmp(argv, "wikipedia.txt")){
								
								for(k=0; k<finalWordLenght; k++){
									asciiSum += (int)finalWord[k];
								}
								
								if(crivoEratostenes(asciiSum)){
									//printf("primo: %d\n", asciiSum);
									primeCount++;																		
								}
								
								*wordPrimeCount = primeCount;
							}
						}
					}
				}
				
			}
		}		
		
		free(*it);		
	}
	
	free(initialWordTokens);
	
	return palindromeCount;
}


int phrasePalindrome(char* buffer, char * argv, int* phrasePrimeCount){
	int i=0, k=0; //laco  
	int palindromeCount=0;
	int primeCount = 0;
	
	
	char** initialPhraseTokens = phraseTokenize(buffer);  
	char** it;
	
	
	
	for(it=initialPhraseTokens; it && *it; ++it){
		
		char** initialWordTokens = wordTokenize(*it);  
		char** it2;
		int initialPhraseLength = 0;
		
		for(it2=initialWordTokens; it2 && *it2; ++it2){
			
			initialPhraseLength+=(int)strlen(*it2);     
		}
		
		char* initialPhrase  = (char*) malloc(sizeof(char)*(initialPhraseLength+1));
		int firstIteration = 1;
		
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
		
		char** finalWordTokens = wordTokenizeDetails(initialPhrase);  
		char** it3;
		int finalPhraseLength = 0;
		
		for(it3=finalWordTokens; it3 && *it3; ++it3){
			finalPhraseLength+=(int)strlen(*it3);
		}
		
		char* finalPhrase  = (char*) malloc(sizeof(char)*(finalPhraseLength+1));
		firstIteration = 1;
		
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
		
		if(finalPhraseLength > 2){
			int equalCount = 0;
			int asciiSum = 0;
			
			for(i=0; i<finalPhraseLength; i++){
				finalPhrase[i] = tolower(finalPhrase[i]);
			}
						
			for(i=0; i<finalPhraseLength; i++){
				if(finalPhrase[i] != finalPhrase[finalPhraseLength-i-1]){
					break;
				} else{
					equalCount++;
					
					if(equalCount == finalPhraseLength){
						
						palindromeCount++;
						
						if(!strcmp(argv, "wikipedia.txt")){
							
							for(k=0; k<finalPhraseLength; k++){
								asciiSum += (int)finalPhrase[k];
							}
							
							if(crivoEratostenes(asciiSum)){
								primeCount++;								
							}
							
							*phrasePrimeCount = primeCount;
						}
						
					}
				}
			}
			
		}				
		
		free(*it);				
	}
	
	free(initialPhraseTokens);
	
	return palindromeCount;
}
