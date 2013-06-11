# I am a comment, and I want to say that the variable CC will be
# the compiler to use.
CC=mpicc
# Hey!, I am comment number 2. I want to say that CFLAGS will be the
# options I'll pass to the compiler.
CFLAGS=-Wall -fopenmp

all: 
	$(CC) $(CFLAGS) t3.c strmap.c -o t3

clean:
	rm -rf *o all
