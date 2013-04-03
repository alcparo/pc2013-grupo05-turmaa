CC     = gcc

CFLAGS = -Wall -lm -lgmp -pthread
INPUT = gauss_legendre_sequencial.c
OUTPUT = gauss_legendre_sequencial

gauss_legendre_paralelo: $(INPUT)
		$(CC)  $(INPUT) -o $(OUTPUT) $(CFLAGS)

clean:
		rm -rf *.o gauss_legendre_sequencial

