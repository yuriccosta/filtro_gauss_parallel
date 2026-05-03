#!/bin/bash

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}   SUPER BENCHMARK AUTOMATIZADO - FILTRO GAUSS      ${NC}"
echo -e "${CYAN}====================================================${NC}"

# ========================================================
# ARRAYS DE PARÂMETROS
# ========================================================
RESOLUCOES=(512 1024 2048)
ITERACOES=(100 500 1000)
THREADS=(2 4 8)
KERNEL=5
# ========================================================

echo "Compilando codigos (O3)..."
gcc -Wall -O3 -fopenmp filtro_gauss.c -o exe_serial -lm
gcc -Wall -O3 -fopenmp filtro_gauss_parallel.c -o exe_paralelo -lm

CSV_FILE="resultados.csv"
 
if [ ! -f "$CSV_FILE" ]; then
    echo "Resolucao,Kernel,Iteracoes,Threads,Tempo" > $CSV_FILE
    echo "Novo arquivo '$CSV_FILE' criado."
else
    echo "Arquivo '$CSV_FILE' ja existe. Adicionando novos dados no final..."
fi
# LAÇO 1: Passa por cada tamanho de imagem
for RES in "${RESOLUCOES[@]}"; do
    IMG_IN="images/image_${RES}x${RES}.pgm"
    
    # Cria a subpasta para a resolução atual (ex: images/512)
    PASTA_OUT="images/${RES}"
    mkdir -p $PASTA_OUT

    if [ ! -f "$IMG_IN" ]; then
        echo -e "${RED}Aviso: Imagem $IMG_IN nao encontrada. Pulando resolucao ${RES}x${RES}...${NC}"
        continue
    fi

    # LAÇO 2: Passa por cada quantidade de iterações
    for ITER in "${ITERACOES[@]}"; do
        echo -e "\n${YELLOW}>>> TESTANDO: Imagem ${RES}x${RES} | Kernel ${KERNEL}x${KERNEL} | Iteracoes: ${ITER} <<<${NC}"
        
        # 1. TESTE SERIAL
        echo -e "${GREEN}[Serial - 1 Thread]${NC} Executando..."
        export OMP_NUM_THREADS=1
        SAIDA_SERIAL=$(./exe_serial $IMG_IN $PASTA_OUT/saida_serial.pgm $KERNEL $ITER | grep "Tempo")
        echo "  -> $SAIDA_SERIAL"

        TEMPO_SERIAL=$(echo $SAIDA_SERIAL | awk '{print $4}')
        echo "${RES},${KERNEL},${ITER},1,${TEMPO_SERIAL}" >> $CSV_FILE

        # 2. TESTES PARALELOS
        for T in "${THREADS[@]}"; do
            echo -e "${GREEN}[Paralelo - ${T} Threads]${NC} Executando..."
            export OMP_NUM_THREADS=$T
            SAIDA_PARALELA=$(./exe_paralelo $IMG_IN $PASTA_OUT/saida_paralela${T}.pgm $KERNEL $ITER | grep "Tempo")
            echo "  -> $SAIDA_PARALELA"


            TEMPO_PARALELO=$(echo $SAIDA_PARALELA | awk '{print $4}')
            echo "${RES},${KERNEL},${ITER},${T},${TEMPO_PARALELA}" >> $CSV_FILE
        done

        # 3. VALIDAÇÃO DE CORRETUDE COM IMAGEMAGICK E DIFF
        echo -n "Validando ${RES}x${RES} com ${ITER} iters: "
        
        NOME_DIFF="$PASTA_OUT/diff_iter${ITER}.png"
        
        # Compara a serial com a paralela de 8 threads
        ERROS=$(compare -metric AE -fuzz 1% $PASTA_OUT/saida_serial.pgm $PASTA_OUT/saida_paralela8.pgm $NOME_DIFF 2>&1)
        
        if [ "$ERROS" -eq "0" ]; then
            echo -e "${GREEN}OK (Correto!)${NC}"
            rm -f $NOME_DIFF
        else
            echo -e "${RED}ERRO ($ERROS pixels divergem)${NC}"
            echo -e "   -> Veja o mapa de erros em: ${YELLOW}$NOME_DIFF${NC}"
        fi

        echo -n "Validando ${RES}x${RES} com ${ITER} iters: "
        if cmp -s "$PASTA_OUT/saida_serial.pgm" "$PASTA_OUT/saida_paralela8.pgm"; then
            echo -e "${GREEN}OK - Identicos bit a bit${NC}"
        else
            echo -e "${RED}ALERTA - Diferenca de ponto flutuante${NC}"
        fi
        
    done
done

echo -e "\n${CYAN}====================================================${NC}"
echo -e "${CYAN} Benchmark Concluído! Resultados gerados com sucesso.${NC}"
echo -e "${CYAN}====================================================${NC}"