CC     = gcc

CFLAGS = -Wall -lm -lgmp -pthread
INPUT = gauss_legendre_paralelo.c
OUTPUT = gauss_legendre_paralelo

gauss_legendre_paralelo: $(INPUT)
		$(CC)  $(INPUT) -o $(OUTPUT) $(CFLAGS)

clean:
		rm -rf *.o gauss_legendre_paralelo

