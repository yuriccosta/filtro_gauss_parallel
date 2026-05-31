#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <ctype.h> 
#include <string.h>

// Função auxiliar para pular comentários no cabeçalho do arquivo PGM
void pgm_skip_comments(FILE *fp) {
    int ch;
    char line[256];
    while ((ch = fgetc(fp)) != EOF && isspace(ch));
    if (ch == '#') {
        if (fgets(line, sizeof(line), fp) == NULL) return; 
        pgm_skip_comments(fp);
    } else {
        fseek(fp, -1, SEEK_CUR);
    }
}

double* read_pgm(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Erro: Nao foi possivel abrir a imagem %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) {
        printf("Erro ao ler assinatura do arquivo.\n");
        exit(EXIT_FAILURE);
    }
    
    if (strcmp(magic, "P5") != 0) {
        printf("Erro: Formato invalido. O arquivo leu: %s. Use PGM P5.\n", magic);
        exit(EXIT_FAILURE);
    }

    pgm_skip_comments(fp);
    if (fscanf(fp, "%d %d", width, height) != 2) {
        printf("Erro ao ler as dimensoes da imagem.\n");
        exit(EXIT_FAILURE);
    }
    
    int maxval;
    pgm_skip_comments(fp);
    if (fscanf(fp, "%d", &maxval) != 1) {
        printf("Erro ao ler o valor maximo de cor.\n");
        exit(EXIT_FAILURE);
    }
    fgetc(fp); 


    double *image = (double *)malloc(*height * *width * sizeof(double));
    unsigned char *row_buffer = (unsigned char *)malloc(*width * sizeof(unsigned char));

    for (int i = 0; i < *height; i++) {
        if (fread(row_buffer, sizeof(unsigned char), *width, fp) != (size_t)(*width)) {
            printf("Erro: Fim de arquivo inesperado lendo os pixels.\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < *width; j++) {
            image[i * (*width) + j] = (double)row_buffer[j]; 
        }
    }

    free(row_buffer);
    fclose(fp);
    return image;
}


void write_pgm(const char *filename, double *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Erro: Nao foi possivel criar o arquivo %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    unsigned char *row_buffer = (unsigned char *)malloc(width * sizeof(unsigned char));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double val = image[i * width + j];
            if (val > 255.0) val = 255.0;
            if (val < 0.0) val = 0.0;
            row_buffer[j] = (unsigned char)(val + 0.5); 
        }
        fwrite(row_buffer, sizeof(unsigned char), width, fp);
    }

    free(row_buffer);
    fclose(fp);
}

void create_gaussian_kernel(int size, double sigma, double *kernel) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = j - size / 2;
            int y = i - size / 2;
            kernel[i * size + j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i * size + j];
        }
    }
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

double pad_edge(double *image, int rows, int cols, int r, int c) {
    if (r < 0) r = 0;
    if (r >= rows) r = rows - 1;
    if (c < 0) c = 0;
    if (c >= cols) c = cols - 1;
    return image[r * cols + c];
}

void apply_convolution(double *image, double *output, int img_height, int img_width, double *kernel, int kernel_size) {
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            double sum = 0.0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    int row = i + k - kernel_size / 2;
                    int col = j + l - kernel_size / 2;
                    double pixel = pad_edge(image, img_height, img_width, row, col);
                    sum += pixel * kernel[k * kernel_size + l];
                }
            }
            output[i * img_width + j] = sum;
        }
    } 



}


double* iterative_gaussian_blur(double *image, int img_height, int img_width, int kernel_size, int iterations) {
    double *kernel = (double *)malloc(kernel_size * kernel_size * sizeof(double));
    create_gaussian_kernel(kernel_size, 1.0, kernel); 

    
    double *output = (double *)malloc(img_height * img_width * sizeof(double));

    for (int i = 0; i < iterations; i++) {
        apply_convolution(image, output, img_height, img_width, kernel, kernel_size);
        
        double *temp = image;
        image = output;
        output = temp;
    }

    
    free(output);
    free(kernel);
    return image;
}

int main(int argc, char *argv[]) {  

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 5) {
        if (rank == 0) {
            printf("Uso incorreto!\n");
            printf("Formato esperado: %s <imagem_entrada> <imagem_saida> <tamanho_kernel> <iteracoes>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int kernel_size = atoi(argv[3]); 
    int iterations = atoi(argv[4]);  

    double *image = NULL;
    int image_width = 0, image_height = 0;

    // Variáveis para MPI_Scatterv
    int *sendcounts = NULL;
    int *displs = NULL;


    if (rank == 0) {
        image = read_pgm(input_file, &image_width, &image_height);
        
        
        sendcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));
         
    }

    MPI_Bcast(&image_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int linhas_por_proc = image_height / num_procs;
    int resto = image_height % num_procs;

    // Para exemplificar e poder construir com base nele
    MPI_Scatterv(image, const int *sendcounts, const int *displs, 
                MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                int root, MPI_Comm comm)





    

    double start_time = MPI_Wtime();
        
    image = iterative_gaussian_blur(image, image_height, image_width, kernel_size, iterations);
    
    double end_time = MPI_Wtime();
    printf("Tempo de execução: %.6f segundos\n", end_time - start_time);
    
    write_pgm(output_file, image, image_width, image_height);
    free(image); 


    MPI_Finalize();
    return 0;
}