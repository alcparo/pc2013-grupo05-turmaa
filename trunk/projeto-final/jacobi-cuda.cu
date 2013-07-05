#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define TPB 128

void printMA(int J_ORDER, float *MA);
void printMB(int J_ORDER, float *MB);
void printX(int J_ORDER, float *X);
void printResults(int ite, int J_ROW_TEST, float result, float MB);

/*
 * 
 */
__global__ void diag(float *dev_MA, float *dev_MB, float *dev_X, int J_ORDER){

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	if(bid < J_ORDER && tid < J_ORDER){
		
		int diagAux;		
		diagAux = dev_MA[bid*J_ORDER+bid];
		
		while(tid<J_ORDER){
			
			if(bid!=tid){
				dev_MA[tid*J_ORDER+bid] = dev_MA[tid*J_ORDER+bid]/diagAux;
			} else{
				dev_MA[tid*J_ORDER+bid] = 0;
			}		
			
			tid += TPB;
		}
		
		__syncthreads();
		dev_MB[bid] = dev_MB[bid] / diagAux;
		dev_X[bid] = dev_MB[bid];			
	}	
}

/*
 * 
 */
__global__ void jacobi(float *dev_MA, float *dev_X, int J_ORDER, float *dev_sum){
	
	int bId = blockIdx.x;
	int tId = threadIdx.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	
	__shared__ float cache[TPB];	
	cache[cacheIndex] = 0;	
	
	if(bId < J_ORDER && tId < J_ORDER){
		
		while(tId < J_ORDER){
			if(bId!=tId){
				temp += (dev_MA[(tId * J_ORDER) + bId] * dev_X[tId]);
			}
			
			tId += TPB;
		}
		
		cache[cacheIndex] = temp;
		__syncthreads();
		
		int i = TPB/2;
		while(i != 0){
			if(cacheIndex < i )
				cache[cacheIndex] += cache[cacheIndex + i];
			
			__syncthreads();
			i /= 2;
		}
		
		if(cacheIndex == 0){
			dev_sum[bId] = cache[0];			
		}
	}
}

int main(int argc, char * argv[]){
			
	//ENTRADA
	FILE * pFile; // Variavel para a abertura do arquivo de entrada.
	/* Abertura do arquivo de entrada */
	if((argc < 2) || (pFile = fopen(argv[1],"r")) == NULL){
		fputs ("File error",stderr); exit (1);
	}
	
	int i,j; // Variavel de controle para as estrutudas de repeticao.
	
	int J_ORDER, J_ROW_TEST;
	float J_ERROR, J_ITE_MAX;	

	fscanf(pFile, "%d%d%f%f", &J_ORDER, &J_ROW_TEST, &J_ERROR, &J_ITE_MAX);
	
	
	float *MA, *MB, *X, *Xold, *MAOriginal, MBOriginal, *sum;		
	MA = 		(float*)malloc(sizeof(float)*J_ORDER*J_ORDER);
	MAOriginal = 	(float*)malloc(sizeof(float)*J_ORDER);	
	MB = 		(float*)malloc(sizeof(float)*J_ORDER);
	X = 		(float*)malloc(sizeof(float)*J_ORDER);
	Xold = 		(float*)malloc(sizeof(float)*J_ORDER);
	sum = 		(float*)malloc(sizeof(float)*J_ORDER);
	
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

///////////////////////////////////fim leitura////////////////////////////////////////////////////////
	

	
	float *dev_MA, *dev_MB, *dev_X, *dev_sum;	
	cudaMalloc( (void**)&dev_MA, J_ORDER*J_ORDER*sizeof(float));
	cudaMalloc( (void**)&dev_MB, J_ORDER*sizeof(float));
	cudaMalloc( (void**)&dev_X, J_ORDER*sizeof(float));
	cudaMalloc( (void**)&dev_sum, J_ORDER*sizeof(float));
	
	cudaMemcpy(dev_MA, MA, J_ORDER*J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MB, MB, J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_X, X, J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
	
	/////// Diagonalização
	
	
	
	diag<<<J_ORDER , TPB>>>(dev_MA, dev_MB, dev_X, J_ORDER); // OK
	cudaMemcpy(MB, dev_MB, J_ORDER*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(X, dev_X, J_ORDER*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	
	int ite = 0;
	float maxDif, maxX, Mr;	
	Mr = FLT_MAX;
		
	while(ite < J_ITE_MAX && Mr > J_ERROR){
		
		jacobi<<<J_ORDER , TPB>>>(dev_MA, dev_X, J_ORDER, dev_sum);
		
		cudaMemcpy(sum, dev_sum, J_ORDER*sizeof(float), cudaMemcpyDeviceToHost);
		
		maxDif = maxX = FLT_MIN;
		for(i = 0 ; i < J_ORDER; i++){
			Xold[i] = X[i];
			X[i] = (MB[i] - sum[i]);
			
			if(fabs(X[i] - Xold[i]) > maxDif)
				maxDif = fabs(X[i] - Xold[i]);
				
			if(fabs(X[i]) > maxX)
				maxX = fabs(X[i]);
		}
		
		Mr = maxDif / maxX;
		
		ite++;
		
		cudaMemcpy(dev_X, X, J_ORDER*sizeof(float), cudaMemcpyHostToDevice);
		
	}
	
	
	float resultAux=0;
	for(j=0; j<J_ORDER; j++){
		resultAux += MAOriginal[j]*X[j];
	}
	
	printResults(ite, J_ROW_TEST, resultAux, MBOriginal);
	
	free(MA); free(MB); free(X); free(Xold); free(MAOriginal); free(sum);
	cudaFree(dev_MA); cudaFree(dev_MB); cudaFree(dev_X); cudaFree(dev_sum);
	
	return 0;
	
}


	
void printMA(int J_ORDER, float *MA){
	int i,j;
	for(i = 0; i<750; i++){
		for(j = 0; j<750; j++){
			printf("%f ", MA[j*J_ORDER+i]);
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