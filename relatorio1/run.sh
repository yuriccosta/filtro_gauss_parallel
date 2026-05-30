gcc -Wall -O3 filtro_gauss.c -lm -fopenmp -o output/run
./output/run images/image_512x512.pgm images/saida_serial.pgm 5 7