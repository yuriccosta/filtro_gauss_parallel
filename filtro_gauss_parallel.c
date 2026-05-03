#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include <ctype.h> // Necessário para a função isspace
#include <string.h>

// Função auxiliar para pular comentários no cabeçalho do arquivo PGM
void pgm_skip_comments(FILE *fp) {
    int ch;
    char line[256];
    while ((ch = fgetc(fp)) != EOF && isspace(ch));
    if (ch == '#') {
        if (fgets(line, sizeof(line), fp) == NULL) return; // Checa o retorno
        pgm_skip_comments(fp);
    } else {
        fseek(fp, -1, SEEK_CUR);
    }
}

// LÊ A IMAGEM PGM E RETORNA A MATRIZ DE DOUBLES
double** read_pgm(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Erro: Nao foi possivel abrir a imagem %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) { // Checa o retorno
        printf("Erro ao ler assinatura do arquivo.\n");
        exit(EXIT_FAILURE);
    }
    
    if (strcmp(magic, "P5") != 0) {
        printf("Erro: Formato de imagem invalido. O arquivo leu: %s. Use PGM P5 (Binario).\n", magic);
        exit(EXIT_FAILURE);
    }

    pgm_skip_comments(fp);
    if (fscanf(fp, "%d %d", width, height) != 2) { // Checa se leu largura E altura (2 valores)
        printf("Erro ao ler as dimensoes da imagem.\n");
        exit(EXIT_FAILURE);
    }
    
    int maxval;
    pgm_skip_comments(fp);
    if (fscanf(fp, "%d", &maxval) != 1) { // Checa o retorno
        printf("Erro ao ler o valor maximo de cor.\n");
        exit(EXIT_FAILURE);
    }
    fgetc(fp); 

    double **image = (double **)malloc(*height * sizeof(double *));
    for (int i = 0; i < *height; i++) {
        image[i] = (double *)malloc(*width * sizeof(double));
    }

    unsigned char *row_buffer = (unsigned char *)malloc(*width * sizeof(unsigned char));
    for (int i = 0; i < *height; i++) {
        // Checa se conseguiu ler a linha inteira da imagem
        if (fread(row_buffer, sizeof(unsigned char), *width, fp) != (size_t)(*width)) {
            printf("Erro: Fim de arquivo inesperado lendo os pixels.\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < *width; j++) {
            image[i][j] = (double)row_buffer[j];
        }
    }

    free(row_buffer);
    fclose(fp);
    return image;
}

// SALVA A MATRIZ DE DOUBLES DE VOLTA EM UM ARQUIVO PGM
void write_pgm(const char *filename, double **image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Erro: Nao foi possivel criar o arquivo %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Escreve o cabeçalho
    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    // Converte os doubles de volta para pixels (0 a 255) e escreve
    unsigned char *row_buffer = (unsigned char *)malloc(width * sizeof(unsigned char));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Garante que o valor não passe de 255 nem fique abaixo de 0
            double val = image[i][j];
            if (val > 255.0) val = 255.0;
            if (val < 0.0) val = 0.0;
            row_buffer[j] = (unsigned char)(val + 0.5); // +0.5 para arredondamento
        }
        fwrite(row_buffer, sizeof(unsigned char), width, fp);
    }

    free(row_buffer);
    fclose(fp);
}



#ifdef _OPENMP
  #include <omp.h>
#else
    #define omp_get_thread_num() 4
    #define omp_get_num_threads() 3
#endif

void create_gaussian_kernel(int size, double sigma, double **kernel){
    if (size % 2 == 0 || size <= 0) {
        printf("O kernel deve ser um número ímpar positivo.\n");
        exit(EXIT_FAILURE);
    }

    int metade = size / 2;
    double sum = 0.0;
    double temp = 2.0 * sigma * sigma;

    // np.arrange e np.meshgrid 
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] = exp(-((i - metade)*(i - metade) + (j - metade)*(j - metade)) / temp);
            sum += kernel[i][j];
        }
    }   

    // Normalizar o kernel np.sum
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }
}


int clamp_int(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

void free_matrix(double **matrix, int rows) {
    if (matrix == NULL) {
        return;
    }

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void apply_convolution(double **image, int image_size_h, int image_size_w, double **kernel, int kernel_size, double **padded_img ,double **output) {
    int pad_h = kernel_size / 2;
    int pad_w = kernel_size / 2;

    int padded_h = image_size_h + 2 * pad_h;
    int padded_w = image_size_w + 2 * pad_w;

    #pragma omp parallel for schedule(static, 2) shared(padded_img, image, image_size_h, image_size_w, padded_h, padded_w, pad_h, pad_w)
    for (int i = 0; i < padded_h; i++) {
        for (int j = 0; j < padded_w; j++) {
            int src_i = clamp_int(i - pad_h, 0, image_size_h - 1);
            int src_j = clamp_int(j - pad_w, 0, image_size_w - 1);
            padded_img[i][j] = image[src_i][src_j];
        }
    }

    // Aplica convolução
    #pragma omp parallel for schedule(dynamic, 2) shared(padded_img, kernel, output, image_size_h, image_size_w, kernel_size)
    for (int i = 0; i < image_size_h; i++){
        for (int j = 0; j < image_size_w; j++){
            double sum = 0.0;
            for (int k = 0; k < kernel_size; k++){
                for (int l = 0; l < kernel_size; l++){
                    sum += padded_img[i + k][j + l] * kernel[k][l];
                }                                    
            }
            output[i][j] = sum;
        }
    }
}


double** iterative_gaussian_blur(double **image, int image_size_h, int image_size_w, int kernel_size, int iterations) {
    
    double **kernel = (double **)malloc(kernel_size * sizeof(double *));
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = (double *)malloc(kernel_size * sizeof(double));
    }
    create_gaussian_kernel(kernel_size, 1.0, kernel);

    // Aloca a padded_img
    int pad_h = kernel_size / 2;
    int pad_w = kernel_size / 2;
    int padded_h = image_size_h + 2 * pad_h;
    int padded_w = image_size_w + 2 * pad_w;
    
    double **padded_img = (double **)malloc(padded_h * sizeof(double *));
    for (int i = 0; i < padded_h; i++) {
        padded_img[i] = (double *)malloc(padded_w * sizeof(double));
    }

    // Aloca o Output Buffer
    double **output = (double **)malloc(image_size_h * sizeof(double *));
    for (int i = 0; i < image_size_h; i++) {
        output[i] = (double *)malloc(image_size_w * sizeof(double));
    }


    for (int iter = 0; iter < iterations; iter++) {
        apply_convolution(image, image_size_h, image_size_w, kernel, kernel_size, padded_img, output);
        double **temp = image;
        image = output;
        output = temp;
    }

    free_matrix(kernel, kernel_size);
    free_matrix(padded_img, padded_h);
    free_matrix(output, image_size_h);

    return image;
}


int main(int argc, char *argv[]) {    
    if (argc != 5) {
        printf("Uso incorreto!\n");
        printf("Formato esperado: %s <imagem_entrada> <imagem_saida> <tamanho_kernel> <iteracoes>\n", argv[0]);
        printf("Exemplo: %s images/entrada.pgm images/saida.pgm 5 100\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int kernel_size = atoi(argv[3]); // Converte texto para número inteiro
    int iterations = atoi(argv[4]);  // Converte texto para número inteiro

    //omp_set_num_threads(8);

    int image_width, image_height;
    // Lê a imagem e preenche automaticamente o width e height
    double **image = read_pgm(input_file, &image_width, &image_height);


    double start_time = omp_get_wtime();
    image = iterative_gaussian_blur(image, image_height, image_width, kernel_size, iterations);
    double end_time = omp_get_wtime();


    printf("Tempo de execução: %.6f segundos\n", end_time - start_time);
    
    write_pgm(output_file, image, image_width, image_height);
    
    // Limpeza da memória final
    free_matrix(image, image_height);
    
    return 0;
}