import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurações visuais dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# 1. Carregar os dados
df_raw = pd.read_csv('resultados_cuda_full.csv')

# Converte colunas para numérico (transforma "N/A" ou "ERRO" em valores nulos do Pandas)
df_raw['Tempo'] = pd.to_numeric(df_raw['Tempo'], errors='coerce')
df_raw['Occupancy'] = pd.to_numeric(df_raw['Occupancy'], errors='coerce')

# 2. Agrupamento e Média (A coluna 'Repeticao' some aqui)
agrupadores = ['Resolucao', 'Iteracoes', 'Kernel', 'BlockSize']
df = df_raw.groupby(agrupadores)[['Tempo', 'Occupancy']].mean().reset_index()

# 3. Separar o baseline (Serial -> BlockSize == 0)
df_serial = df[df['BlockSize'] == 0][['Resolucao', 'Iteracoes', 'Kernel', 'Tempo']]
df_serial = df_serial.rename(columns={'Tempo': 'T_Serial'})

# 4. Filtrar apenas os resultados paralelos (CUDA -> BlockSize > 0)
df_cuda = df[df['BlockSize'] > 0].copy()

# 5. Mesclar e calcular o Speedup e Eficiência
df_cuda = df_cuda.merge(df_serial, on=['Resolucao', 'Iteracoes', 'Kernel'], how='inner')
df_cuda['Speedup'] = df_cuda['T_Serial'] / df_cuda['Tempo']

# Em CUDA, a Eficiência solicitada é a própria Ocupância média obtida
df_cuda = df_cuda.rename(columns={'Occupancy': 'Eficiencia_Ocupancia'})

# 6. Salvar o novo CSV tratado
saida_csv = Path('graph/resultados_metricas_cuda.csv')
saida_csv.parent.mkdir(parents=True, exist_ok=True)
df_cuda.to_csv(saida_csv, index=False)
print(f"CSV de métricas gerado: {saida_csv}")

# ==========================================
# GERAÇÃO DOS GRÁFICOS
# ==========================================
def gera_grid_cuda(dataframe, valor_y, titulo, nome_arquivo, label_y):
    g = sns.relplot(
        data=dataframe, 
        x="BlockSize",       # Eixo X agora é o tamanho do bloco
        y=valor_y, 
        hue="Resolucao",     # Cores diferentes para cada resolução
        col="Iteracoes",     # Colunas para as iterações
        row="Kernel",        # Linhas para o tamanho da máscara
        kind="line", marker="o", palette="tab10", height=4, aspect=1.2,
        facet_kws={'sharey': False}, linewidth=2
    )
    
    for ax in g.axes.flat:
        # Define as marcações do eixo X exatamente para os blocos testados
        ax.set_xticks(sorted(dataframe['BlockSize'].unique()))
        ax.set_ylabel(label_y)
            
    g.fig.suptitle(titulo, y=1.02, fontweight='bold', fontsize=16)
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
    print(f"Gráfico gerado: {nome_arquivo}")
    plt.close()

# Gerar os comparativos
gera_grid_cuda(df_cuda, "Tempo", "Tempo de Execução CUDA: BlockSize vs Resolução", "graph/tempo_cuda.png", "Tempo (s)")
gera_grid_cuda(df_cuda, "Speedup", "Speedup (Serial vs CUDA): BlockSize vs Resolução", "graph/speedup_cuda.png", "Speedup")
gera_grid_cuda(df_cuda, "Eficiencia_Ocupancia", "Eficiência/Ocupância da GPU: BlockSize vs Resolução", "graph/ocupancia_cuda.png", "Ocupância Ativa (%)")