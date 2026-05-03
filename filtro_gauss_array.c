#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
#endif

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

double pad_edge (double *image, int rows, int cols, int r, int c) {
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


int main() {
    int img_height = 5, img_width = 5;
    int kernel_size = 5;
    int iterations = 1000000;
    double sigma = 1.0;

    double image[25] = {
        1, 2, 3, 2, 1,
        2, 4, 6, 4, 2,
        3, 6, 9, 6, 3,
        2, 4, 6, 4, 2,
        1, 2, 3, 2, 1
    };

    double *kernel = malloc(kernel_size * kernel_size * sizeof(double));
    double *aux = malloc(img_height * img_width * sizeof(double));
    double *output = malloc(img_height * img_width * sizeof(double));

        for (int i = 0; i < 25; i++) 
        {
            aux[i] = image[i];
        }

    create_gaussian_kernel(kernel_size, sigma, kernel);

    for (int i = 0; i < iterations; i++) {
        apply_convolution(aux, output, img_height, img_width, kernel, kernel_size);
        for (int j = 0; j < 25; j++) 
        {
            aux[j] = output[j];
        }
    }

    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            printf("%lf ", aux[i * img_width + j]);
        }
        printf("\n");
    }

    free(kernel); free(aux); free(output);
    return 0;
}