# -*- coding: utf-8 -*-
"""
Script de Pré-processamento e Filtragem de Dados de Expressão Proteica.

Este script é projetado para:
1. Ler dados brutos de ficheiros Excel (promotores, RBS, constructs combinados).
2. Aplicar etapas de filtragem para remover amostras de baixa qualidade.
3. Opcionalmente, filtrar outliers com base nas métricas de IQR.
4. Padronizar/escalar a coluna de expressão proteica usando StandardScaler e RobustScaler,
    e guardar os scalers para uso posterior.
5. Construir as sequências de DNA finais combinando promotores, um 'spacer' e RBS.
6. Preparar o dataset final com as colunas essenciais.
7. Exportar o dataset processado para ficheiros CSV e Excel para uso em etapas posteriores
    (como o treino de modelos de Machine Learning).
"""

# --- 0. Importações de Bibliotecas ---
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
import sys
import re

# --- 1. Configurações Globais ---

# O diretório base é o diretório onde este script está localizado.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório para os ficheiros de dados (inputs e outputs).
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True) # Garante que a pasta 'data' existe

# Diretório para guardar os scalers.
SCALER_DIR = os.path.join(BASE_DIR, 'scalers')
os.makedirs(SCALER_DIR, exist_ok=True) # Garante que a pasta 'scalers' existe

# Caminhos relativos dos datasets de entrada dentro da pasta 'data'.
dataset_constructs_path = os.path.join(DATA_DIR, "promoters_RBS.xlsx")
dataset_promoters_path = os.path.join(DATA_DIR, "promoters.xlsx")
dataset_rbs_path = os.path.join(DATA_DIR, "RBS.xlsx")

# Suprimir avisos específicos do openpyxl sobre cabeçalhos/rodapés
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --- 2. Leitura dos Dados ---
print(f"A carregar dados dos ficheiros Excel na pasta: {DATA_DIR}...")
try:
    df_constructs = pd.read_excel(dataset_constructs_path)
    promoters_df = pd.read_excel(dataset_promoters_path)
    rbs_df = pd.read_excel(dataset_rbs_path)
    print("Dados carregados com sucesso.")
except FileNotFoundError as e:
    print(f"ERRO: Um dos ficheiros de dados não foi encontrado. Verifique a pasta '{DATA_DIR}'.")
    print(f"Erro: {e}")
    sys.exit(1) # Sai do script se os ficheiros de dados não forem encontrados
except Exception as e:
    print(f"ERRO inesperado ao carregar dados: {e}")
    sys.exit(1) # Sai do script em caso de outros erros de carregamento

# --- 3. Filtragem Inicial ---
print("A aplicar filtragem inicial...")
# Lista de flags indicadoras de baixa qualidade a serem removidas
bad_flags_columns = ["bad.promo", "bad.prot", "bad.DNA", "bad.RNA", "min.prot", "max.prot", "min.RNA"]

# Verificação se as colunas de flags existem antes de filtrar
missing_flags = [flag for flag in bad_flags_columns if flag not in df_constructs.columns]
if missing_flags:
    print(f"AVISO: As seguintes colunas de flag não foram encontradas no dataset de construtos e serão ignoradas na filtragem: {', '.join(missing_flags)}")
    # Remove as flags ausentes da lista para não causar erro na condição
    bad_flags_columns = [flag for flag in bad_flags_columns if flag not in missing_flags]

# Aplica a filtragem para manter apenas as linhas onde todas as flags são False
df_processed = df_constructs[~(df_constructs[bad_flags_columns]).any(axis=1)].copy()

# Remover todas as linhas com NA em 'prot' (sem dados de expressão)
df_processed.dropna(subset=['prot'], inplace=True)

# Resultados da filtragem inicial
print(f"Número de linhas no dataset original: {len(df_constructs)}")
print(f"Número de linhas no dataset após filtragem inicial: {len(df_processed)}")
print(f"Percentagem de linhas removidas na filtragem inicial: {100-(len(df_processed)/len(df_constructs))*100:.2f}%")


# --- 4. Filtragem Opcional de Outliers (IQR) ---
# Defina 'True' para ativar a filtragem de outliers, 'False' para desativar.
outliers_filter = False

if outliers_filter:
    print("\nA aplicar filtragem de outliers usando o método IQR...")
    initial_rows_for_outliers = len(df_processed)

    # Função auxiliar para aplicar filtragem IQR
    def apply_iqr_filter(df, column_name):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    df_processed = apply_iqr_filter(df_processed, 'prot').copy() # .copy() importante
    df_processed = apply_iqr_filter(df_processed, 'count.DNA').copy() # .copy() importante
    df_processed = apply_iqr_filter(df_processed, 'count.RNA').copy() # .copy() importante

    # Reiniciar índices por conveniência
    df_processed.reset_index(drop=True, inplace=True)

    # Estatísticas da filtragem de outliers
    print(f"Número de linhas antes da filtragem de outliers: {initial_rows_for_outliers}")
    print(f"Número de linhas após filtragem de outliers: {len(df_processed)}")
    removed_perc = 100 - (len(df_processed) / initial_rows_for_outliers) * 100
    print(f"Percentagem de linhas removidas na filtragem de outliers: {removed_perc:.2f}%")
else:
    print("\nFiltragem de outliers (IQR) está desativada ('outliers_filter = False').")

# --- 5. Estandardização da Expressão Proteica ---
print("\nA aplicar escalagem dos dados de expressão proteica ('prot')...")

# Normalização com StandardScaler
scaler_std = StandardScaler()
df_processed['prot_z'] = scaler_std.fit_transform(df_processed[['prot']])

# Normalização com RobustScaler
scaler_rob = RobustScaler()
df_processed['prot_scaled'] = scaler_rob.fit_transform(df_processed[['prot']])

# Guardar os scalers para uso posterior
joblib.dump(scaler_std, os.path.join(SCALER_DIR, 'scaler_standard.pkl'))
joblib.dump(scaler_rob, os.path.join(SCALER_DIR, 'scaler_robust.pkl'))
print(f"Scalers guardados em: {SCALER_DIR}")
print("Escalagem concluída (colunas 'prot_z' e 'prot_scaled' criadas).")




# --- 6. Preparação do Dataset Final (Construção de Sequências) ---
# --- Definir spacer ---
spacer = "AACTT"  # Spacer neutro e curto, sem codões de start ou stop

# --- Remover Sequências de Reconhecimento de Enzimas de Restrição (RE Sites) ---.

RE_SITES_TO_REMOVE_FROM_PROMOTER = [re.escape("GGCGCGCC")] # AscI site com o espaço no fim

RE_SITES_TO_REMOVE_FROM_RBS = [re.escape("CATATG")] # NdeI site com o espaço no início


# Função auxiliar para aplicar a remoção de padrões
def clean_sequence_re_sites(sequence, patterns_to_remove, is_start=True):
    seq_str = str(sequence) # Garante que é string
    for pattern in patterns_to_remove:
        if is_start: # Remove do início da string
            seq_str = re.sub(f"^{pattern}", "", seq_str)
        else: # Remove do fim da string
            seq_str = re.sub(f"{pattern}$", "", seq_str)
    return seq_str.strip() # .strip() remove quaisquer espaços restantes no início/fim

# Aplica a remoção dos RE sites ANTES da remoção geral de espaços e aspas.
promoters_df['Sequence'] = promoters_df['Sequence'].apply(lambda x: clean_sequence_re_sites(x, RE_SITES_TO_REMOVE_FROM_PROMOTER, is_start=True))
rbs_df['Sequence'] = rbs_df['Sequence'].apply(lambda x: clean_sequence_re_sites(x, RE_SITES_TO_REMOVE_FROM_RBS, is_start=False))

# Eliminar quaisquer espaços internos remanescentes (se existirem na sequência principal) e aspas duplas literais
promoters_df['Sequence'] = promoters_df['Sequence'].str.replace(" ", "").str.replace('"', '', regex=False)
rbs_df['Sequence'] = rbs_df['Sequence'].str.replace(" ", "").str.replace('"', '', regex=False)

# --- Merge com sequências dos promotores ---
df_merged = df_processed.merge(
    promoters_df[['full.name', 'Sequence']],
    left_on='Promoter.TTL',
    right_on='full.name',
    how='left'
)
df_merged = df_merged.rename(columns={'Sequence': 'promoter_seq'}).drop(columns='full.name')

# Remover quaisquer aspas duplas literais das sequências de promotores
# antes da concatenação. Isso garante que a sequência final não as contenha.
df_merged['promoter_seq'] = df_merged['promoter_seq'].str.replace('"', '', regex=False)


# --- Merge com sequências dos RBS ---
df_merged = df_merged.merge(
    rbs_df[['full.name', 'Sequence']],
    left_on='RBS.TTL',
    right_on='full.name',
    how='left'
)
df_merged = df_merged.rename(columns={'Sequence': 'rbs_seq'}).drop(columns='full.name')

# Remover quaisquer aspas duplas literais das sequências de RBS
# antes da concatenação.
df_merged['rbs_seq'] = df_merged['rbs_seq'].str.replace('"', '', regex=False)


# Construção da sequência final (Promotor + Spacer + RBS)
df_merged['full_sequence'] = df_merged['promoter_seq'] + spacer + df_merged['rbs_seq']

# Selecionar apenas as colunas essenciais
df_minimal = df_merged[['Promoter', 'RBS', 'target', 'full_sequence', 'prot', 'prot_z', 'prot_scaled']].rename(columns={
    'full_sequence': 'sequence'
})


# --- 7. Exportar para Ficheiros ---
# Os ficheiros de output serão guardados na subpasta 'data'.
output_training_csv = os.path.join(DATA_DIR, "training_data.csv")
output_final_excel = os.path.join(DATA_DIR, "training_data.xlsx")

print(f"\nA exportar datasets processados para a pasta: {DATA_DIR}...")
df_minimal.to_csv(output_training_csv, index=False)
df_minimal.to_excel(output_final_excel, index=False)
print(f"Dataset CSV guardado em: {output_training_csv}")
print(f"Dataset Excel guardado em: {output_final_excel}")

print("\nScript de pré-processamento concluído com sucesso.")
