Palindromo Sequencial

      Para compilar:
      gcc palindromo_sequencial.c -o palindromo_sequencial -Wall -lm -fopenmp

      Para executar:
      ./palindromo_sequencial <arquivo_de_entrada>

Palindromo Paralelo openMP

      Para compilar:
      gcc palindromo_openmp.c -o palindromo_openmp -Wall -lm -fopenmp

      Para executar:
      ./palindromo_openmp <arquivo_de_entrada> <numero_de_threads>

Palindromo Paralelo MPI

      Para compilar:
      mpicc palindromo_mpi.c

      Para executar:
      mpirun -np 5 --host node06,node02,node03,node04,node05 <arquivo_de_saida.out> <arquivo_de_entrada>

