CC = gcc
CFLAGS = -g

SRC_DIR = ./src
BUILD_DIR = ./build
BIN_DIR = ./bin

all: $(SRC_DIR)/mnist-digits.c $(SRC_DIR)/driver.c $(SRC_DIR)/libbmp/libbmp.c
	$(CC) $(CFLAGS) $(SRC_DIR)/mnist-digits.c $(SRC_DIR)/driver.c $(SRC_DIR)/libbmp/libbmp.c -o mnist-digits.out