import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import (
    f_oneway,
    friedmanchisquare,
    kruskal,
    levene,
    mannwhitneyu,
    shapiro,
    ttest_ind,
    ttest_rel,
    wilcoxon,
    kstest, 
    normaltest, 
    anderson,
)
import numpy as np
from scipy.stats import norm
from collections import namedtuple




# TABEALA DE DISTRIBUIÇÃO DE FREQUENCIA:

def tabela_distribuicao_frequencias(dataframe, coluna, coluna_frequencia=False):
    """Cria uma tabela de distribuição de frequências para uma coluna de um dataframe.
    Espera uma coluna categórica.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe com os dados.
    coluna : str
        Nome da coluna categórica.
    coluna_frequencia : bool
        Informa se a coluna passada já é com os valores de frequência ou não. Padrão: False

    Returns
    -------
    pd.DataFrame
        Dataframe com a tabela de distribuição de frequências.
    """ 
    
    df_estatistica = pd.DataFrame()

    if coluna_frequencia:
        df_estatistica["frequencia"] = dataframe[coluna]
        df_estatistica["frequencia_relativa"] = df_estatistica["frequencia"] / df_estatistica["frequencia"].sum()
    else:
        df_estatistica["frequencia"] = dataframe[coluna].value_counts().sort_index()
        df_estatistica["frequencia_relativa"] = dataframe[coluna].value_counts(normalize=True).sort_index()
        
    df_estatistica["frequencia_acumulada"] = df_estatistica["frequencia"].cumsum()
    df_estatistica["frequencia_relativa_acumulada"] = df_estatistica["frequencia_relativa"].cumsum()
    
    return df_estatistica



# HISTOGRAMA / BOXPLOT

def composicao_histograma_boxplot(dataframe, coluna, intervalos="auto"):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={
            "height_ratios": (0.15, 0.85),
            "hspace": 0.02
        }
    )
    
    sns.boxplot(
        data=dataframe,
        x=coluna,
        showmeans=True,
        meanline=True,
        meanprops={"color": "C1", "linewidth": 1.5, "linestyle": "--"},
        medianprops={"color": "C2", "linewidth": 1.5, "linestyle": "--"},
        ax=ax1,
    )
    
    sns.histplot(data=dataframe, x=coluna, kde=True, bins=intervalos, ax=ax2)
    
    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", color="gray", alpha=0.5)
        ax.set_axisbelow(True)
    
    ax2.axvline(dataframe[coluna].mean(), color="C1", linestyle="--", label="Média")
    ax2.axvline(dataframe[coluna].median(), color="C2", linestyle="--", label="Mediana")
    ax2.axvline(dataframe[coluna].mode()[0], color="C3", linestyle="--", label="Moda")
    
    ax2.legend()
    
    plt.show()



# TESTE DE SHAPIRO-WILK

def analise_shapiro(dataframe, alfa=0.05):
    print("Teste de Shapiro-Wilk")
    for coluna in dataframe.columns:
        estatistica_sw, valor_p_sw = shapiro(dataframe[coluna], nan_policy="omit")
        print(f"{estatistica_sw=:.3f}")
        if valor_p_sw > alfa:
            print(f"{coluna} segue uma distribuição normal (valor p: {valor_p_sw:.3f})")
        else:
            print(f"{coluna} não segue uma distribuição normal (valor p: {valor_p_sw:.3f})")
        


# TESTE DE LEVENE

def analise_levene(dataframe, alfa=0.05, centro="mean"):
    print("Teste de Levene")
    
    estatistica_levene, valor_p_levene = levene(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        center=centro, 
        nan_policy="omit"
    )
        
    print(f"{estatistica_levene=:.3f}")
    if valor_p_levene > alfa:
        print(f"Variâncias iguais (valor p: {valor_p_levene:.3f})")
    else:
        print(f"Ao menos uma variância é diferente (valor p: {valor_p_levene:.3f})")



# TESTE DE SHAPIRO / LEVENE

def analises_shapiro_levene(dataframe, alfa=0.05, centro="mean"):
    analise_shapiro(dataframe, alfa)

    print()

    analise_levene(dataframe, alfa, centro)

def teste_z(dados, media_pop, desvio_pop):
    print("Teste Z")
    n = len(dados)
    media_amostral = np.mean(dados)
    desvio_media_amostral = desvio_pop / np.sqrt(n)
    z = (media_amostral - media_pop) / desvio_media_amostral
    valor_p = 1 - norm.cdf(z)
    TesteZ = namedtuple("TesteZ", ["estatistica", "valor_p"])
    return TesteZ(z, valor_p)




# TESTE T DE STUDENT (AMOSTRAS INDEPENDENTES)

def analise_ttest_ind(
    dataframe,
    alfa=0.05,
    variancias_iguais=True,
    alternativa="two-sided",
):
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy="omit"
    )
        
    print(f"{estatistica_ttest=:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")




# TESTE T DE STUDENT (AMOSTRAS PAREADAS)

def analise_ttest_rel(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_rel(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        alternative=alternativa,
        nan_policy="omit"
    )
        
    print(f"{estatistica_ttest=:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")




# TESTE DE ANÁLISE DE VARIÂNCIA

def analise_anova_one_way(
    dataframe,
    alfa=0.05,
):
    
    print("Teste ANOVA one way")
    estatistica_f, valor_p_f = f_oneway(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        nan_policy="omit"
    )
        
    print(f"{estatistica_f=:.3f}")
    if valor_p_f > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_f:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_f:.3f})")





# TESTE DE WILCOXON (DUAS AMOSTRAS PAREADAS)

def analise_wilcoxon(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):

    print("Teste de Wilcoxon")
    estatistica_wilcoxon, valor_p_wilcoxon = wilcoxon(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa,
    )

    print(f"{estatistica_wilcoxon=:.3f}")
    if valor_p_wilcoxon > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_wilcoxon:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_wilcoxon:.3f})")





# TESTE DE MANN WHITNEY (DUAS AMOSTRAS INDEPENDENTES)

def analise_mannwhitneyu(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):

    print("Teste de Mann-Whitney")
    estatistica_mw, valor_p_mw = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa,
    )

    print(f"{estatistica_mw=:.3f}")
    if valor_p_mw > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_mw:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_mw:.3f})")





# TESTE DE FRIEDMAN (3 OU MAIS AMOSTRAS PAREADAS)

def analise_friedman(
    dataframe,
    alfa=0.05,
):

    print("Teste de Friedman")
    estatistica_friedman, valor_p_friedman = friedmanchisquare(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
    )

    print(f"{estatistica_friedman=:.3f}")
    if valor_p_friedman > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_friedman:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_friedman:.3f})")






# TESTE DE KRUSKAL (3 OU MAIS AMOSTRAS INDEPENDENTES)

def analise_kruskal(
    dataframe,
    alfa=0.05,
):

    print("Teste de Kruskal")
    estatistica_kruskal, valor_p_kruskal = kruskal(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
    )

    print(f"{estatistica_kruskal=:.3f}")
    if valor_p_kruskal > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_kruskal:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_kruskal:.3f})")





# TESTE DE DISTRIBUIÇÃO NORMAL (KS-TESTE / NORMALIDADE / ANDERSON)

def testar_normalidade(df, excluir_colunas=['Time', 'Class']):
    """
    Testa a normalidade das colunas numéricas de um DataFrame usando os testes KS, D'Agostino-Pearson e Anderson-Darling.
    
    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    excluir_colunas (list): Lista de colunas a serem excluídas da análise.
    
    Retorna:
    pd.DataFrame: DataFrame com os resultados dos testes de normalidade.
    """
    colunas_numericas = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluir_colunas]
    resultados = {}
    
    for col in colunas_numericas:
        dados = df[col].dropna()  # Remover valores NaN
        
        if len(dados) > 8:  # Normaltest requer pelo menos 8 valores
            dados_norm = (dados - dados.mean()) / dados.std()
            
            ks_p = kstest(dados_norm, 'norm')[1]
            dagostino_p = normaltest(dados)[1]
            anderson_stat = anderson(dados, 'norm').statistic
            
            normal = (ks_p > 0.05) and (dagostino_p > 0.05)
            
            resultados[col] = {
                'KS_p': ks_p,
                "D'Agostino_p": dagostino_p,
                'Anderson_Stat': anderson_stat,
                'Normal': normal
            }
    
    df_resultados = pd.DataFrame(resultados).T
    return df_resultados.sort_values(by="Anderson_Stat")