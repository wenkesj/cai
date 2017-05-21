CC ?= gcc
CFLAGS = -fPIC -Wall -Wextra -O2 -g
LDFLAGS = -shared
RM = rm -f
TARGET_LIB = libweakai.dylib weakai.o
SRC_PATH = ./weakai
SRCS = $(shell find $(SRC_PATH) -name '*.c' | sort -k 1nr | cut -f2-)
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)
