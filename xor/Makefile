CC ?= gcc
RM = rm -f
BIN_NAME = xor.o
SRCS = xor.c
WEAKAI = ../

.PHONY: all
all:
	$(CC) -L$(WEAKAI) -lweakai -I$(WEAKAI) -o $(BIN_NAME) $(SRCS)

.PHONY: clean
clean:
	-${RM} ${BIN_NAME}
