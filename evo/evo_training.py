#%%
# -*- coding: utf-8 -*-
"""
Script de Treino de SVM para Regressão usando Embeddings EVO.

Este script é projetado para:
1. Carregar dados de sequência de DNA e valores de expressão proteica.
2. Utilizar o modelo EVO como extrator de features para gerar embeddings de sequência.
3. Treinar e otimizar um modelo Support Vector Regressor (SVR) com validação cruzada.
4. Avaliar o desempenho do SVR com métricas detalhadas.
5. Guardar o modelo treinado, os resultados da avaliação e os embeddings gerados.

"""

# --- Importações de Bibliotecas ---
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch
import os
import sys
import gc # Módulo útil para gerir a memória
import datetime # Para timestamps nos ficheiros de saída
import joblib # Para guardar o modelo SVM
from tqdm import tqdm # Para barras de progresso na extração de features


# Importar a classe Evo do pacote 'evo-model'.
try:
    from evo.models import Evo
    print("Módulo 'evo.models.Evo' importado com sucesso.")
except ImportError:
    print("ERRO: O módulo 'evo.models.Evo' não foi encontrado. Verifique que 'evo-model' está instalado corretamente no ambiente.")
    sys.exit(1) # Sair se a biblioteca essencial não estiver disponível

"""
Ativar computação TensorFloat32 (TF32) para speedups em GPUs NVIDIA compatíveis.
TF32 acelera as operações de matriz em GPUs Ampere+ sem perda significativa de precisão
para a maioria dos casos de uso de machine learning.
"""
torch.backends.cuda.matmul.allow_tf32 = True
print("TensorFloat32 ativado para operações de matriz CUDA (se compatível).")


# --- 1. Configurações Globais do Script ---

## Configuração do Utilizador ##
# O diretório base é o diretório onde este script está localizado.
# Isso torna a estrutura da pasta portátil.
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) # Caminho dinâmico para a pasta do script

# Diretório para guardar os resultados (modelos, métricas, embeddings).
# Será criada uma subpasta 'results' dentro do BASE_DATA_DIR.
RESULTS_OUTPUT_BASE_DIR = os.path.join(BASE_DATA_DIR, 'results') # ALTERADO: Subpasta 'results'
os.makedirs(RESULTS_OUTPUT_BASE_DIR, exist_ok=True) # Garante que o diretório de resultados existe

# Caminho completo para o dataset de treino (com sequências e target 'prot_scaled').
# Assume que 'training_data.csv' está na mesma pasta que este script.
TRAINING_DATA_FILE = os.path.join(BASE_DATA_DIR, 'training_data.csv') # ALTERADO: Caminho relativo ao script

# Caminho completo para o ficheiro de embeddings EVO (será criado ou carregado aqui).
# Será guardado na subpasta 'results'.
EVO_EMBEDDINGS_FILE = os.path.join(RESULTS_OUTPUT_BASE_DIR, 'evo_embeddings.npy') # ALTERADO: Guardado em 'results'

# Nome do modelo EVO a ser carregado do Hugging Face Hub.
# Opções comuns: "evo-1-8k-base", "evo-1.5-8k-base", "evo-1-131-base".
# O evo-1-8k-base foi usado nas suas configurações anteriores.
EVO_MODEL_NAME = "evo-1-8k-base"

# Tamanho do lote (batch size) para a extração de features do EVO.
# Um valor maior acelera a inferência na GPU, mas requer mais VRAM.
# Ajuste com base na VRAM disponível na GPU.
# EVO-1-8k-base tem ~6.45 mil milhões de parâmetros, requer ~12.9 GiB (FP16/BF16)
# para os pesos, mais memória para ativações.
EVO_BATCH_SIZE = 1 # Manter este valor se a VRAM for limitada.

# Definir o dispositivo de execução (GPU ou CPU).
# Dá prioridade ao GPU se disponível para aceleração computacional.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDispositivo selecionado para execução: {device}.")
if device == 'cpu':
    print("AVISO: GPU não disponível. A execução na CPU será significativamente mais lenta.")


# --- 2. Carregar os Dados ---
# Carrega o dataset de treino, que contém as sequências de DNA e os valores de expressão.

try:
    df = pd.read_csv(TRAINING_DATA_FILE)
    print(f"\nDados carregados com sucesso de {TRAINING_DATA_FILE}. Dimensão: {df.shape}")

except FileNotFoundError:
    print(f"ERRO: O arquivo de dados '{TRAINING_DATA_FILE}' não foi encontrado.")
    sys.exit(1)
except ValueError as e:
    print(f"ERRO nos dados: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERRO inesperado ao carregar os dados: {e}")
    sys.exit(1)


# Extrai as sequências e os valores de expressão para uso no pipeline.
sequences = df['sequence'].tolist()
prot_scaled = df['prot_scaled'].values # 'prot_scaled' representa a variável alvo escalada com RobustScaler.


# --- 3. Função para Extrair Features do EVO ---
def extract_evo_features(sequences: list, model, tokenizer, device: str, batch_size: int) -> np.ndarray:
    """
    Extrai embeddings de sequência usando o modelo EVO.
    Estes embeddings servem como features de entrada para o modelo SVM.
    O modelo EVO retorna (logits, hidden_states), e a média dos hidden_states
    da última camada é usada como o embedding da sequência.

    Args:
        sequences (list): Lista de sequências de DNA.
        model: O modelo EVO carregado.
        tokenizer: O tokenizer do modelo EVO.
        device (str): Dispositivo para execução ('cuda' ou 'cpu').
        batch_size (int): Tamanho do lote para inferência.

    Returns:
        np.ndarray: Array NumPy contendo os embeddings extraídos para todas as sequências.
    """
    print(f"\nExtraindo features do EVO para {len(sequences)} sequências com batch_size={batch_size}...")
    model.eval() # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    all_embeddings = []

    # Processa as sequências em lotes para gerir eficientemente a memória.
    # Usar tqdm para uma barra de progresso.
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extraindo embeddings EVO"):
        batch_sequences = sequences[i:i + batch_size]

        # Tokeniza e adiciona padding às sequências do lote.
        input_ids_batch = [tokenizer.encode(seq) for seq in batch_sequences]

        # Garante padding uniforme para o lote (se EVO.encode não fizer isso por si só)
        max_batch_len = max(len(ids) for ids in input_ids_batch)
        padded_input_ids_batch = [ids + [tokenizer.pad_id] * (max_batch_len - len(ids)) for ids in input_ids_batch]


        # Converte a lista de IDs de token para um tensor PyTorch e move para o dispositivo.
        input_ids = torch.tensor(padded_input_ids_batch, dtype=torch.long).to(device)

        with torch.no_grad(): # Desativa o cálculo de gradientes para economizar memória e acelerar a inferência.
            # Realiza a inferência com o modelo EVO.
            logits, hidden_states = model(input_ids)

            # Calcula o embedding da sequência fazendo a média dos hidden_states da última camada.
            sequence_embeddings = hidden_states[0].mean(dim=1)

            # Move os embeddings para a CPU e converte para NumPy array.
            all_embeddings.append(sequence_embeddings.cpu().numpy())

        # Força a recolha de lixo e esvazia a cache da GPU para liberar memória após cada lote.
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Empilha todos os embeddings processados em um único array NumPy.
    return np.vstack(all_embeddings)


# --- 4. Carregar Modelo EVO e Extrair Features ---
# Esta secção gere o carregamento do modelo EVO e a obtenção das features.
evo_features = None

# Tenta carregar embeddings pré-calculados do disco para economizar tempo.
print(f"\nVerificando se existem embeddings EVO pré-calculados em: {EVO_EMBEDDINGS_FILE}...")
if os.path.exists(EVO_EMBEDDINGS_FILE):
    try:
        loaded_embeddings = np.load(EVO_EMBEDDINGS_FILE)
        # Verifica se os embeddings carregados correspondem ao dataset atual.
        if loaded_embeddings.shape[0] == len(sequences):
            evo_features = loaded_embeddings
            print(f"Carregando embeddings do EVO de {EVO_EMBEDDINGS_FILE}. Dimensão: {evo_features.shape}")
        else:
            print(f"Aviso: Embeddings carregados ({loaded_embeddings.shape[0]} sequências) não correspondem ao dataset atual ({len(sequences)} sequências). Recalculando.")
    except Exception as e:
        print(f"Erro ao carregar embeddings de {EVO_EMBEDDINGS_FILE}: {e}. Recalculando.")

# Se os embeddings não forem carregados com sucesso (ou não existirem/estiverem desatualizados), inicializa o modelo EVO e extrai-os.
if evo_features is None:
    print(f"\nIniciando o carregamento/extração de features do EVO '{EVO_MODEL_NAME}'...")

    # Instancia o modelo EVO. Este passo pode descarregar os pesos do modelo
    # do Hugging Face Hub se ainda não estiverem em cache local.
    print(f"Passo 4.1: Instanciando o modelo Evo('{EVO_MODEL_NAME}'). Pode descarregar os pesos do modelo.")
    evo_model_instance = Evo(EVO_MODEL_NAME)
    model, tokenizer = evo_model_instance.model, evo_model_instance.tokenizer

    print(f"Passo 4.2: Modelo Evo instanciado. Número de Parâmetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} Bilhões.")
    print(f"Passo 4.3: Movendo o modelo para o dispositivo '{device}'...")

    # Move o modelo para o dispositivo selecionado (GPU ou CPU).
    # Para GPUs, usa half precision (FP16) para economizar VRAM.
    if device.startswith('cuda'):
        print("Carregando o modelo em half precision (FP16) na GPU para otimizar VRAM...")
        model.half().to(device)
    else:
        model.to(device) # Mover para CPU RAM

    print("Passo 4.4: Modelo EVO carregado e pronto.")
    if device == 'cpu':
        print("AVISO: O modelo está a ser executado na CPU. A inferência será significativamente mais lenta.")
    else:
        print("O modelo está a ser executado na GPU para maior velocidade.")

    print(f"Passo 4.5: Iniciando a extração de features com batch_size={EVO_BATCH_SIZE}...")
    evo_features = extract_evo_features(sequences, model, tokenizer, device, batch_size=EVO_BATCH_SIZE)

    # Garante que o diretório de destino para os embeddings existe antes de salvar.
    os.makedirs(os.path.dirname(EVO_EMBEDDINGS_FILE), exist_ok=True) # Cria o dir de embeddings
    np.save(EVO_EMBEDDINGS_FILE, evo_features)
    print(f"\nPasso 4.6: Embeddings do EVO gerados com sucesso e salvos em {EVO_EMBEDDINGS_FILE}.")


# --- 5. Preparar Dados para o SVM ---
# Prepara os dados para o treinamento do modelo SVM.
X = evo_features  # Features de entrada para o SVM (embeddings do EVO)
y = prot_scaled   # Variável alvo (expressão génica), que deve estar na escala escalada ('prot_scaled').

print(f"\nDimensão de X (features do EVO): {X.shape}")
print(f"Dimensão de y (target 'prot_scaled' - escalado): {y.shape}")


# --- 6. Treinar o SVM (SVR - Support Vector Regressor) ---
# Implementa a validação cruzada e a otimização de hiperparâmetros para o SVR.

# Configuração da Validação Cruzada K-Fold (5 folds).
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # random_state para reprodutibilidade das divisões.

# Definição do espaço de busca para otimização de hiperparâmetros do SVR.
# modificar se desejado
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

svr = SVR() # Inicializa o modelo Support Vector Regressor.

print("\nIniciando busca por hiperparâmetros com GridSearchCV...")
grid_search = GridSearchCV(svr, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X, y) # Executa a busca pelos melhores hiperparâmetros.

print(f"\nMelhores hiperparâmetros encontrados: {grid_search.best_params_}")
best_svr_model = grid_search.best_estimator_ # O modelo SVR com os melhores hiperparâmetros.


# --- 7. Avaliar o Modelo SVM ---
# Avalia o desempenho do modelo SVR otimizado usando validação cruzada.
print("\nAvaliação com validação cruzada nos melhores parâmetros...")
mse_scores = []
mae_scores = []
r2_scores = []
pearson_scores = []

# Itera sobre cada fold da validação cruzada.
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    fold_model = SVR(**grid_search.best_params_) # Re-instancia o SVR com os melhores params
    fold_model.fit(X_train, y_train)
    y_pred = fold_model.predict(X_test) # Faz previsões no conjunto de teste do fold.

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    try:
        pearson_corr, _ = pearsonr(y_test, y_pred)
    except ValueError:
        pearson_corr = np.nan

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    pearson_scores.append(pearson_corr)

    print(f"Fold {fold+1}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Pearson={pearson_corr:.4f}")

print("\n--- Resultados Finais da Validação Cruzada (Médias e Desvios Padrão) ---")
final_cv_metrics = {
    "MSE médio": f"{np.nanmean(mse_scores):.4f} +/- {np.nanstd(mse_scores):.4f}",
    "MAE médio": f"{np.nanmean(mae_scores):.4f} +/- {np.nanstd(mae_scores):.4f}",
    "R2 médio": f"{np.nanmean(r2_scores):.4f} +/- {np.nanstd(r2_scores):.4f}",
    "Correlação de Pearson média": f"{np.nanmean(pearson_scores):.4f} +/- {np.nanstd(pearson_scores):.4f}"
}
for metric, value in final_cv_metrics.items():
    print(f"{metric}: {value}")


print("\n--- Métricas no Dataset Completo (com o melhor modelo treinado uma vez no dataset inteiro) ---")
final_predictions = best_svr_model.predict(X)

global_mse = mean_squared_error(y, final_predictions)
global_mae = mean_absolute_error(y, final_predictions)
global_r2 = r2_score(y, final_predictions)
try:
    global_pearson, _ = pearsonr(y, final_predictions)
except ValueError:
    global_pearson = np.nan

global_metrics = {
    "MSE Global": f"{global_mse:.4f}",
    "MAE Global": f"{global_mae:.4f}",
    "R2 Global": f"{global_r2:.4f}",
    "Pearson Global": f"{global_pearson:.4f}"
}
for metric, value in global_metrics.items():
    print(f"{metric}: {value}")


# --- 8. Guardar Resultados e Modelo ---
os.makedirs(RESULTS_OUTPUT_BASE_DIR, exist_ok=True)

results_filename = os.path.join(RESULTS_OUTPUT_BASE_DIR, "evo_svm_results_robust_scaled.txt")
model_save_filename = os.path.join(RESULTS_OUTPUT_BASE_DIR, "best_evo_svm_model_robust_scaled.joblib")
predictions_data_filename = os.path.join(RESULTS_OUTPUT_BASE_DIR, f"evo_svm_predictions_data_robust_scaled.npy")


print(f"\n--- A guardar resultados em: {results_filename} ---")
with open(results_filename, 'w') as f:
    f.write(f"Resultados de Treinamento EVO-SVM para Regressão (Target RobustScaler)\n")
    f.write(f"----------------------------------------------------\n\n")

    f.write(f"Configurações Iniciais:\n")
    f.write(f"  Modelo EVO: {EVO_MODEL_NAME}\n")
    f.write(f"  Batch Size EVO: {EVO_BATCH_SIZE}\n")
    f.write(f"  Dispositivo: {device}\n")
    f.write(f"  Dataset: {os.path.basename(TRAINING_DATA_FILE)}\n")
    f.write(f"  Número de Amostras: {X.shape[0]}\n")
    f.write(f"  Número de Features (Embeddings EVO): {X.shape[1]}\n")
    f.write(f"  Folds de Validação Cruzada: {n_splits}\n")
    f.write(f"\n")

    f.write(f"Melhores Hiperparâmetros SVR Encontrados:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\n")

    f.write(f"Resultados da Validação Cruzada (Médias e Desvios Padrão - Escala Escalada com RobustScaler):\n")
    for metric, value in final_cv_metrics.items():
        f.write(f"  {metric}: {value}\n")
    f.write(f"\n")

    f.write(f"Métricas no Dataset Completo (Com o melhor modelo treinado uma vez no dataset inteiro - Escala Escalada com RobustScaler):\n")
    for metric, value in global_metrics.items():
        f.write(f"  {metric}: {value}\n")
    f.write(f"\n")

print(f"Resultados guardados com sucesso em {results_filename}.")

joblib.dump(best_svr_model, model_save_filename)
print(f"Melhor modelo SVR guardado em: {model_save_filename}")

predictions_data = np.vstack((y, final_predictions)).T
np.save(predictions_data_filename, predictions_data)
print(f"Valores reais e previstos (escala escalada) guardados em {predictions_data_filename}.")

print("\nScript de treinamento EVO-SVM concluído.")
