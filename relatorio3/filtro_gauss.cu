#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h> 
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// Tamanho máximo 25 suporta até um kernel 5x5.
#define MAX_MASK_SIZE 25
__constant__ double d_kernel[MAX_MASK_SIZE];

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

// Cria a máscara no Host (CPU)
void create_gaussian_kernel(int size, double sigma, double *kernel) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = j - size / 2;
            int y = i - size / 2;
            kernel[i * size + j] = expf(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i * size + j];
        }
    }
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}



__global__ void apply_convolution_cuda(double *d_image, double *d_output, int img_height, int img_width, int kernel_size) {
    // Calcula a linha e coluna globais a partir dos índices de bloco e thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Apenas threads dentro dos limites da imagem trabalham
    if (row < img_height && col < img_width) {
        double sum = 0.0;
        int half_k = kernel_size / 2;

        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int r = row + k - half_k;
                int c = col + l - half_k;

                // Tratamento de bordas (Clamping/Padding diretamente na GPU)
                if (r < 0) r = 0;
                if (r >= img_height) r = img_height - 1;
                if (c < 0) c = 0;
                if (c >= img_width) c = img_width - 1;

                // Lê o pixel da memória global e a máscara da memória constante
                double pixel = d_image[r * img_width + c];
                sum += pixel * d_kernel[k * kernel_size + l];
            }
        }
        d_output[row * img_width + col] = sum;
    }
}



int main(int argc, char *argv[]) {  
    if (argc != 6) {
        printf("Uso incorreto!\n");
        printf("Formato esperado: %s <imagem_entrada> <imagem_saida> <tamanho_kernel> <iteracoes> <tamanho_bloco>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int kernel_size = atoi(argv[3]); 
    int iterations = atoi(argv[4]); 
    int block_size = atoi(argv[5]);

    if (kernel_size * kernel_size > MAX_MASK_SIZE) {
        printf("Erro: tamanho de kernel muito grande. Max suportado: 5x5\n");
        return EXIT_FAILURE;
    }

    // Leitura da imagem para a RAM (Host)
    int img_width = 0, img_height = 0;
    double *h_image = read_pgm(input_file, &img_width, &img_height);
    size_t img_bytes = img_width * img_height * sizeof(double);

    // Criação da Máscara Gaussiana na RAM (Host)
    double *h_kernel = (double *)malloc(kernel_size * kernel_size * sizeof(double));
    create_gaussian_kernel(kernel_size, 1.0, h_kernel); 

    // Alocação na VRAM (Device)
    double *d_image_in, *d_image_out;
    cudaMalloc((void**)&d_image_in, img_bytes);
    cudaMalloc((void**)&d_image_out, img_bytes);

    // Transferência dos dados (Host -> Device)
    cudaMemcpy(d_image_in, h_image, img_bytes, cudaMemcpyHostToDevice);
    // Transfere a máscara para a MEMÓRIA CONSTANTE da GPU
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(double));

    // Definição da grade e blocos para o kernel
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((img_width + dimBlock.x - 1) / dimBlock.x, 
                 (img_height + dimBlock.y - 1) / dimBlock.y);

    // Preparar cronometragem CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Iniciando %d iteracoes na GPU...\n", iterations);
    cudaEventRecord(start); // Marca o inicio do tempo

    for (int i = 0; i < iterations; i++) {
        apply_convolution_cuda<<<dimGrid, dimBlock>>>(d_image_in, d_image_out, img_height, img_width, kernel_size);
        
        double *temp = d_image_in;
        d_image_in = d_image_out;
        d_image_out = temp;
    }

    cudaEventRecord(stop); // Marca o fim do tempo
    cudaEventSynchronize(stop); // Aguarda o fim de todas as iterações

    // Cálculo do tempo
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tempo de execucao (CUDA): %.6f segundos\n", ms / 1000.0f);

    // Como os ponteiros foram invertidos, d_image_in sempre tem a última saída válida
    cudaMemcpy(h_image, d_image_in, img_bytes, cudaMemcpyDeviceToHost);
    
    // Escreve o resultado no arquivo final
    write_pgm(output_file, h_image, img_width, img_height);

    // Limpeza da VRAM e RAM
    cudaFree(d_image_in);
    cudaFree(d_image_out);
    free(h_image);
    free(h_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}