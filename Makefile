CC = gcc
CFLAGS = -g -I/usr/local/include
LDFLAGS = -L/usr/local/lib/ -lgsl -lcblas -lm

SRC_DIR = ./src

all: driver.o mnist-digits.o neural-network.o libbmp.o
	$(CC) $(LDFLAGS) driver.o mnist-digits.o neural-network.o libbmp.o -o driver.out

driver.o: $(SRC_DIR)/driver.c
	$(CC) $(CFLAGS) -c $(SRC_DIR)/driver.c

mnist-digits.o: $(SRC_DIR)/mnist-digits.c $(SRC_DIR)/mnist-digits.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/mnist-digits.c

neural-network.o: $(SRC_DIR)/neural-network.c $(SRC_DIR)/neural-network.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/neural-network.c

libbmp.o: $(SRC_DIR)/libbmp/libbmp.c
	$(CC) $(CFLAGS) -c $(SRC_DIR)/libbmp/libbmp.c 

clean: 
	rm *.o