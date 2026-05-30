CC = mpicc
CFLAGS = -Wall -O3
LDFLAGS = -lm
TARGET = filtro_mpi
SRC = filtro_mpi.c

# Parâmetros padrão para o teste rápido
NP = 4
IMG_IN = images/image_512x512.pgm
IMG_OUT = images/512/image_512x512.pgm
KERNEL = 5
ITERS = 1000

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(IMG_IN) $(IMG_OUT) $(KERNEL) $(ITERS)

clean:
	rm -f $(TARGET) saida_mpi.pgm
