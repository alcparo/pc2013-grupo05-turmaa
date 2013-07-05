#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

void printMA(int J_ORDER, float **MA);
void printMB(int J_ORDER, float *MB);
void printX(int J_ORDER, float *X);
void printResults(int ite, int J_ROW_TEST, float result, float MB);

int main(int argc, char * argv[]){
	
	int J_ORDER, J_ROW_TEST;
	float J_ERROR, J_ITE_MAX;
	
	int i,j; // Variavel de controle para as estrutudas de repeticao.
		
	//ENTRADA
	FILE * pFile; // Variavel para a abertura do arquivo de entrada.
	/* Abertura do arquivo de entrada */
	if((argc < 2) || (pFile = fopen(argv[1],"r")) == NULL){
		fputs ("File error",stderr); exit (1);
	}
	
	fscanf(pFile, "%d%d%f%f", &J_ORDER, &J_ROW_TEST, &J_ERROR, &J_ITE_MAX);
	
	float **MA, *MAOriginal, *MB, MBOriginal, *X;	
	
	MA = 		(float**)malloc(sizeof(float*)*J_ORDER);
	MB = 		(float*)malloc(sizeof(float)*J_ORDER);
	MAOriginal = 	(float*)malloc(sizeof(float)*J_ORDER);
	X = 		(float*)malloc(sizeof(float)*J_ORDER);
		
	for(i = 0; i<J_ORDER; i++){
		
		MA[i] = (float*)malloc(sizeof(float)*J_ORDER);
		
		for(j = 0; j<J_ORDER; j++){
			fscanf(pFile, "%f", &MA[i][j]);
			if(i == J_ROW_TEST)
				MAOriginal[j] = MA[i][j];
		}
	}
		
	for(i=0; i<J_ORDER; i++){
		fscanf(pFile, "%f", &MB[i]);
		if(i == J_ROW_TEST)
			MBOriginal = MB[i];		
	}
	
	fclose(pFile);
	
	int diagAux;
	for(i = 0; i<J_ORDER; i++){
		diagAux = MA[i][i];
		for(j = 0; j<J_ORDER; j++){
			if(i!=j){
				MA[i][j] = MA[i][j]/diagAux;
			} else{
				MA[i][j] = 0;
			}
		}
		
		MB[i] = MB[i] / diagAux;
		X[i] = MB[i];
	}	
	
	int ite=0;
	float Mr = FLT_MAX;

	float sumAux[J_ORDER];
	float maxDif, maxX, xAux;	

	while((ite<J_ITE_MAX) && (Mr > J_ERROR)){
		
		maxDif = maxX = FLT_MIN;
		
		for(i = 0; i<J_ORDER; i++){
			
			sumAux[i]=0;
		
			for(j = 0; j<J_ORDER; j++){
				if(i!=j)
					sumAux[i] += (MA[i][j]*X[j]);
			}		
			
		}	
		
		for(i = 0; i <J_ORDER; i++){
			
			xAux = X[i];
			X[i] = MB[i] - sumAux[i];
				
			if(fabs(X[i] - xAux) > maxDif)
				maxDif = fabs(X[i] - xAux);
			
			if(fabs(X[i]) > maxX)
				maxX = fabs(X[i]);
		
		}		
		
		Mr = maxDif / maxX;		
		ite++;	
	}

	float resultAux=0;
	for(j=0; j<J_ORDER; j++){
		resultAux += MAOriginal[j]*X[j];
	}

	printResults(ite, J_ROW_TEST, resultAux, MBOriginal);
	
	free(MA);
	free(MB);
	free(X);
	free(MAOriginal);
	
	return 0;	
}

	
void printMA(int J_ORDER, float **MA){
	int i,j;
	for(i = 0; i<J_ORDER; i++){
		for(j = 0; j<J_ORDER; j++){
			printf("%f ", MA[i][j]);
		}
		printf("\n");
	}	
}

void printMB(int J_ORDER, float *MB){
	int i;
	for(i = 0; i<J_ORDER; i++){		
		printf("%f ", MB[i]);		
		printf("\n");
	}	
}

void printX(int J_ORDER, float *X){
	int i;
	for(i = 0; i<J_ORDER; i++){		
		printf("%f ", X[i]);		
		
	}	
	printf("\n");
}

void printResults(int ite, int J_ROW_TEST, float result, float MB){
	printf("\n\n---------------------------------------------------------\n"
	"Iterations: %d\n"
	"RowTest: %d => [%f] =? %f\n"
	"---------------------------------------------------------\n\n", ite, J_ROW_TEST, result, MB);
}