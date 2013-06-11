      Para compilar:

	make
ou
	mpicc t3.c strmap.c -o t3 -fopenmp

      Para executar:

      mpirun -np <numero de nós> --hostfile hosts ./t3 filtrado.txt <numero de threads>

