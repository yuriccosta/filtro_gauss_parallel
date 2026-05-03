#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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

void apply_convolution(double **image, int image_size_h, int image_size_w, double **kernel, int kernel_size, double **output) {
    int pad_h = kernel_size / 2;
    int pad_w = kernel_size / 2;

    int padded_h = image_size_h + 2 * pad_h;
    int padded_w = image_size_w + 2 * pad_w;

    double **padded_img = (double **)malloc(padded_h * sizeof(double *));


    for (int i = 0; i < padded_h; i++) {
        padded_img[i] = (double *)malloc(padded_w * sizeof(double));
        for (int j = 0; j < padded_w; j++) {
            int src_i = clamp_int(i - pad_h, 0, image_size_h - 1);
            int src_j = clamp_int(j - pad_w, 0, image_size_w - 1);
            padded_img[i][j] = image[src_i][src_j];
        }
    }

    // Aplica convolução
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

    free_matrix(padded_img, padded_h);
}


double** iterative_gaussian_blur(double **image, int image_size_h, int image_size_w, int kernel_size, int iterations) {
    
    double **kernel = (double **)malloc(kernel_size * sizeof(double *));
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = (double *)malloc(kernel_size * sizeof(double));
    }
    create_gaussian_kernel(kernel_size, 1.0, kernel);

    for (int iter = 0; iter < iterations; iter++) {
        double **output = (double **)malloc(image_size_h * sizeof(double *));
        for (int i = 0; i < image_size_h; i++) {
            output[i] = (double *)malloc(image_size_w * sizeof(double));
        }
        apply_convolution(image, image_size_h, image_size_w, kernel, kernel_size, output);
        free_matrix(image, image_size_h);
        image = output;
    }
    free_matrix(kernel, kernel_size);

    return image;
}


int main() {
    double image_data[5][5] = {
    {1, 2, 3, 2, 1},
    {2, 4, 6, 4, 2},
    {3, 6, 9, 6, 3},
    {2, 4, 6, 4, 2},
    {1, 2, 3, 2, 1}};

   // Alocação dinâmica da imagem inicial para que o free_matrix não quebre
    double **image = (double **)malloc(5 * sizeof(double *));
    for (int i = 0; i < 5; i++) {
        image[i] = (double *)malloc(5 * sizeof(double));
        for (int j = 0; j < 5; j++) {
            image[i][j] = image_data[i][j];
        }
    }


    image = iterative_gaussian_blur(image, 5, 5, 5, 1000000);
    

    printf("Blur aplicado com sucesso!\n");
    printf("Matriz:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", image[i][j]);
        }
        printf("\n");
    }
    free_matrix(image, 5);
    
    return 0;
}