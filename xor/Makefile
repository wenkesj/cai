CC ?= gcc
RM = rm -f
BIN_NAME = xor.o
SRCS = xor.c
CAI = ../

.PHONY: all
all:
	$(CC) -L$(CAI) -lcai -I$(CAI) -o $(BIN_NAME) $(SRCS)

.PHONY: clean
clean:
	-${RM} ${BIN_NAME}
