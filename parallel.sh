gcc -Wall -O3 filtro_gauss_parallel.c -lm -fopenmp -o output/parallel_run
./output/parallel_run images/image_512x512.pgm images/saida_parallel.pgm 5 100