Gauss-Legendre

      Versão Sequencial
      Para compilar:
      gcc gauss_legendre_sequencial.c -o gauss_legendre_sequencial -Wall -lm -lgmp

      Para executar:
      ./gauss_legendre_sequencial

      Versão Paralela
      Para compilar:
      gcc gauss_legendre_paralelo.c -o gauss_legendre_paralelo -Wall -lm -lgmp -pthread

      Para executar:
      ./gauss_legendre_paralelo

Borwein

      Versão Sequencial
      Para compilar:
      gcc borwein_sequencial.c -o borwein_sequencial -lgmp -lm

      Para executar:
      ./borwein_sequencial

      Versão Paralela
      Para compilar:
      gcc borwein_paralelo.c -o borwein_paralelo -lgmp -lpthread -lm

      Para executar:
      ./borwein_paralelo


Monte Carlo

      Versão Sequencial
      Para compilar:
      gcc monte_carlo_sequencial.c -o monte_carlo_sequencial -lm -lgmp 

      Para executar:
      ./monte_carlo_sequencial

      Versão Paralela
      Para compilar:
      gcc monte_carlo_paralelo.c -o monte_carlo_paralelo -lm -lgmp -pthread

      Para executar:
      ./monte_carlo_paralelo
