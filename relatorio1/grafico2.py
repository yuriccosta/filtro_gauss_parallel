import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

df_raw = pd.read_csv('resultados.csv')

# 1. Agrupamento incluindo o Kernel
agrupadores = ['Resolucao', 'Iteracoes', 'Threads', 'Kernel']
df = df_raw.groupby(agrupadores)['Tempo'].mean().reset_index()

# 2. Cálculo de Speedup e Eficiência
df_t1 = df[df['Threads'] == 1][['Resolucao', 'Iteracoes', 'Kernel', 'Tempo']].rename(columns={'Tempo': 'T1'})
df = df.merge(df_t1, on=['Resolucao', 'Iteracoes', 'Kernel'])
df['Speedup'] = df['T1'] / df['Tempo']
df['Eficiencia'] = df['Speedup'] / df['Threads']


def gera_grid_completo(dataframe, valor_y, titulo, nome_arquivo, label_y, hline=None, ideal_line=False):
    g = sns.relplot(
        data=dataframe, x="Threads", y=valor_y, hue="Resolucao", 
        col="Iteracoes", row="Kernel",
        kind="line", marker="o", palette="tab10", height=4, aspect=1.2,
        facet_kws={'sharey': False}, linewidth=2
    )
    
    for ax in g.axes.flat:
        ax.set_xticks([1, 2, 4, 8])
        if ideal_line:
            threads = sorted(dataframe['Threads'].unique())
            ax.plot(threads, threads, color='black', linestyle='--', alpha=0.5, label='Ideal')
        if hline:
            ax.axhline(hline, color='black', linestyle='--', alpha=0.5)
            
    g.fig.suptitle(titulo, y=1.02, fontweight='bold', fontsize=16)
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
    print(f"Grafico gerado: {nome_arquivo}")

# Gerar os comparativos
gera_grid_completo(df, "Tempo", "Análise de Tempo: Resolução vs Iteração vs Kernel", "graph/tempo_kernel.png", "Tempo (s)")
gera_grid_completo(df, "Speedup", "Análise de Speedup: Resolução vs Iteração vs Kernel", "graph/speedup_kernel.png", "Speedup ($S$)", ideal_line=True)
gera_grid_completo(df, "Eficiencia", "Análise de Eficiência: Resolução vs Iteração vs Kernel", "graph/eficiencia_kernel.png", "Eficiência ($E$)", hline=1.0)