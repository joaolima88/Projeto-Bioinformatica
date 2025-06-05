# -*- coding: utf-8 -*-

"""
Script de Treinamento de SVM para Regressão.
Este script utiliza o modelo EVO como extrator de features para otimizar
construções de DNA para expressão génica.
"""

# --- 0. Importações de Bibliotecas ---
# pré-instalar as bibliotecas no ambiente Python adequado.
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch
import os
import sys
import gc  # Módulo útil para gerir a memória

# Importar a classe Evo.
try:
    from evo.models import Evo

    print("Módulo 'evo.models.Evo' importado com sucesso.")
except ImportError:
    print(
        "ERRO: O módulo 'evo.models.Evo' não foi encontrado. Verifique que se 'evo-model' está instalado corretamente no ambiente.")
    sys.exit(1)  # Sair se a biblioteca essencial não estiver disponível

# Ativar computação TensorFloat32 para speedups em GPUs NVIDIA compatíveis.
# Pode acelerar as operações de matriz sem perda significativa de precisão.
torch.backends.cuda.matmul.allow_tf32 = True
print("TensorFloat32 ativado para operações de matriz CUDA.")

# --- 1. Configurações Iniciais ---
# Definir o dispositivo de execução (GPU ou CPU).
# Dá prioridade ao GPU se disponível para aceleração computacional.
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"\ndispositivo = {device}.")
if device == 'cpu':
    print("AVISO: GPU não disponível. A execução na CPU será significativamente mais lenta.")

# Caminho completo para o dataset de treino.
TRAINING_DATA_FILE = os.path.expanduser('~/projeto/data/training_data.csv')

# Diretório para salvar e carregar os embeddings do EVO.
# Permite reutilizar embeddings se estes já foram pré-calculados, economiza tempo em execuções futuras.
DATA_PATH_FOR_EMBEDDINGS = os.path.dirname(TRAINING_DATA_FILE)
EVO_EMBEDDINGS_FILE = os.path.join(DATA_PATH_FOR_EMBEDDINGS, 'evo_embeddings.npy')

"""
Nome do modelo EVO a ser carregado.
Outros modelos disponíveis: evo-1.5-8k-base, evo-1-131-base, evo-1-8k-crispr e evo-1-8k-transposon
aqui só nos interessa o evo-1-8k-base, evo-1.5-8k-base e o evo-1-131-base
"""
EVO_MODEL_NAME = "evo-1-8k-base"

"""
Tamanho do lote (batch size) para a extração de features do EVO.
O batch size afeta principalmente a quantidade de memória necessária 
para as ativações durante a inferência.
Um valor maior acelera a inferência na GPU, mas requer mais VRAM.
Ajustar o valor com base na VRAM disponível na GPU
O evo-1-8k-base tem ~ 6.45 mil milhões de parâmetros. 
mesmo em precisão mista (FP16/BF16), ele ocupa ~ 12.9 GiB de VRAM apenas para os pesos
mínimo recomendado seria 16-24 GiB (para além dos 12.9 GiB precisa-mos de memória para o PyTorch)
"""
EVO_BATCH_SIZE = 1

# --- 2. Carregar os Dados ---
# Carrega o dataset de treino, que contém as sequências de DNA e os valores de expressão.
try:
    df = pd.read_csv(TRAINING_DATA_FILE)
    print(f"\nDados carregados com sucesso de {TRAINING_DATA_FILE}. Dimensão: {df.shape}")
except FileNotFoundError:
    print(f"ERRO: O arquivo de dados '{TRAINING_DATA_FILE}' não foi encontrado.")
    print(f"Verifique o caminho do arquivo.")
    sys.exit(1)  # Sair se o arquivo não for encontrado

# Verifica se o DataFrame contém as colunas esperadas ('target', 'sequence', 'prot_z').
# 'target' é o ID do construct, 'sequence' é a sequência de DNA, e 'prot_z' é a expressão proteica normalizada.
if not all(col in df.columns for col in ['target', 'sequence', 'prot_z']):
    print("ERRO: colunas inesperadas no Dataframe.")
    sys.exit(1)

# Extrai as sequências e os valores de expressão para uso no pipeline.
sequences = df['sequence'].tolist()
prot_z = df['prot_z'].values


# --- 3. Função para Extrair Features do EVO ---
def extract_evo_features(sequences: list, model, tokenizer, device: str, batch_size: int):
    """
    Extrai embeddings de sequência usando o modelo EVO.
    Estes embeddings servem como features de entrada para o modelo SVM.
    O modelo EVO retorna (logits, hidden_states), e a média dos hidden_states
    da última camada é usada como o embedding da sequência.
    """
    print(f"\nExtraindo features do EVO para {len(sequences)} sequências com batch_size={batch_size}...")
    model.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    all_embeddings = []

    # Processa as sequências em lotes para gerenciar eficientemente a memória.
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]

        # Encontra o comprimento máximo da sequência no lote para aplicar padding.
        max_seq_length = max(len(seq) for seq in batch_sequences)
        input_ids_batch = []
        for seq in batch_sequences:
            # Tokeniza a sequência de DNA.
            tokenized_seq = tokenizer.tokenize(seq)
            # Adiciona padding ao final para igualar o comprimento do lote.
            padding_needed = max_seq_length - len(tokenized_seq)
            padded_tokenized_seq = tokenized_seq + [tokenizer.pad_id] * padding_needed
            input_ids_batch.append(padded_tokenized_seq)

        # Converte a lista de IDs de token para um tensor PyTorch e move para o dispositivo.
        input_ids = torch.tensor(input_ids_batch, dtype=torch.long).to(device)

        with torch.no_grad():  # Desativa o cálculo de gradientes para economizar memória e acelerar a inferência.
            # Realiza a inferência com o modelo EVO.
            logits, hidden_states = model(input_ids)

            # Calcula o embedding da sequência fazendo a média dos hidden_states da última camada.
            # hidden_states[0] geralmente contém a saída da última camada antes do layer norm final.
            sequence_embeddings = hidden_states[0].mean(dim=1)

            # Move os embeddings para a CPU e converte para NumPy array.
            all_embeddings.append(sequence_embeddings.cpu().numpy())

        print(f"  Processado {min(i + batch_size, len(sequences))}/{len(sequences)} sequências...")
        # Força a recolha de lixo e esvazia a cache da GPU para liberar memória.
        gc.collect()
        if device == 'cuda:0':
            torch.cuda.empty_cache()

    # Empilha todos os embeddings processados em um único array NumPy.
    return np.vstack(all_embeddings)


# --- 4. Carregar Modelo EVO e Extrair Features ---
# Esta secção gere o carregamento do modelo EVO e a obtenção das features.
evo_features = None

# Tenta carregar embeddings pré-calculados do disco para economizar tempo.
if os.path.exists(EVO_EMBEDDINGS_FILE):
    try:
        loaded_embeddings = np.load(EVO_EMBEDDINGS_FILE)
        # Verifica se os embeddings carregados correspondem ao dataset atual.
        if loaded_embeddings.shape[0] == len(sequences):
            evo_features = loaded_embeddings
            print(f"\nCarregando embeddings do EVO de {EVO_EMBEDDINGS_FILE}. Dimensão: {evo_features.shape}")
        else:
            print(
                f"\nAviso: Embeddings carregados ({loaded_embeddings.shape[0]} sequências) não correspondem ao dataset atual ({len(sequences)} sequências). Recalculando.")
    except Exception as e:
        print(f"\nErro ao carregar embeddings de {EVO_EMBEDDINGS_FILE}: {e}. Recalculando.")

# Se os embeddings não forem carregados, inicializa o modelo EVO e extrai-os.
if evo_features is None:
    print(f"\nA iniciar o EVO '{EVO_MODEL_NAME}'...")
    try:
        # Instancia o modelo EVO. vai iniciar o download dos pesos do modelo
        # do Hugging Face Hub se ainda não estiverem em cache local.
        print(f"Passo 4.1: Instanciando o modelo Evo('{EVO_MODEL_NAME}'). A descarregar os pesos do modelo.")
        evo_model_instance = Evo(EVO_MODEL_NAME)
        model, tokenizer = evo_model_instance.model, evo_model_instance.tokenizer

        print(
            f"Passo 4.2: Modelo Evo instanciado. Parâmetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} Bilhões.")
        print(f"Passo 4.3: A mover o modelo para o dispositivo '{device}'...")

        # Move o modelo para o dispositivo selecionado (GPU ou CPU).
        # Usa half precision (FP16) na GPU para economizar VRAM.
        if device == 'cuda:0':
            print("A carregar o modelo em half precision...")
            model.half().to(device)
        else:
            model.to(device)  # Mover para CPU RAM

        print("Passo 4.4: Modelo EVO carregado com sucesso.")
        if device == 'cpu':
            print("AVISO: O modelo está a ser executado na CPU. A inferência será lenta.")
        else:
            print("O modelo está a ser executado na GPU.")

        print(f"Passo 4.5: Iniciando a extração de features com batch_size={EVO_BATCH_SIZE}...")
        evo_features = extract_evo_features(sequences, model, tokenizer, device, batch_size=EVO_BATCH_SIZE)

        # Garante que o diretório de destino para os embeddings existe.
        os.makedirs(DATA_PATH_FOR_EMBEDDINGS, exist_ok=True)
        np.save(EVO_EMBEDDINGS_FILE, evo_features)
        print(f"\nPasso 4.6: Embeddings do EVO gerados com sucesso e salvos em {EVO_EMBEDDINGS_FILE}.")

    except Exception as e:
        print(f"\nERRO CRÍTICO na secção 4: Falha ao carregar ou extrair features do modelo EVO: {e}")
        print(
            "Verifique a sua conexão com a internet (para download do modelo) e se o 'evo-model' está instalado corretamente.")
        print("Se o problema persistir, pode haver um problema de configuração do ambiente ou de recursos inesperado.")
        sys.exit(1)  # Sair em caso de erro crítico no carregamento/extração do EVO

# --- 5. Preparar Dados para o SVM ---
# Prepara os dados para o treinamento do modelo SVM.
X = evo_features  # Features de entrada para o SVM (embeddings do EVO)
y = prot_z  # Variável alvo (expressão génica)

print(f"\nDimensão de X (features do EVO): {X.shape}")
print(f"Dimensão de y (target 'prot_z'): {y.shape}")

# --- 6. Treinar o SVM (SVR - Support Vector Regressor) ---
# Implementa a validação cruzada e a otimização de hiperparâmetros para o SVR.

# Configuração da Validação Cruzada K-Fold (5 folds).
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Definição do espaço de busca para otimização de hiperparâmetros do SVR.
# Os ranges podem ser expandidos para explorar mais combinações de hiperparâmetros.
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],  # Parâmetro de regularização
    'gamma': ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],  # Parâmetro do kernel RBF
    'kernel': ['rbf']  # Tipo de kernel (RBF é uma escolha comum para dados não lineares)
}

svr = SVR()  # Inicializa o modelo Support Vector Regressor.

print("\nIniciando busca por hiperparâmetros com GridSearchCV...")
# GridSearchCV realiza uma busca exaustiva por todas as combinações de hiperparâmetros.
# 'n_jobs=-1' utiliza todos os núcleos de CPU disponíveis para paralelizar a busca.
grid_search = GridSearchCV(svr, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X, y)  # Executa a busca pelos melhores hiperparâmetros.

print(f"\nMelhores hiperparâmetros encontrados: {grid_search.best_params_}")
best_svr_model = grid_search.best_estimator_  # O modelo SVR com os melhores hiperparâmetros.

# --- 7. Avaliar o Modelo ---
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

    # Treina o modelo com os melhores parâmetros no conjunto de treino do fold atual.
    best_svr_model.fit(X_train, y_train)
    y_pred = best_svr_model.predict(X_test)  # Faz previsões no conjunto de teste do fold.

    # Calcula as métricas de avaliação para o fold atual.
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Tenta calcular a correlação de Pearson, tratando casos de input constante.
    try:
        pearson_corr, _ = pearsonr(y_test, y_pred)
    except ValueError:
        pearson_corr = np.nan  # Define como NaN se a correlação não for definida.

    # Armazena as métricas do fold.
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    pearson_scores.append(pearson_corr)

    print(f"Fold {fold + 1}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Pearson={pearson_corr:.4f}")

print("\n--- Resultados Finais da Validação Cruzada ---")
# Apresenta a média e o desvio padrão das métricas em todos os folds.
print(f"MSE médio: {np.nanmean(mse_scores):.4f} +/- {np.nanstd(mse_scores):.4f}")
print(f"MAE médio: {np.nanmean(mae_scores):.4f} +/- {np.nanstd(mae_scores):.4f}")
print(f"R2 médio: {np.nanmean(r2_scores):.4f} +/- {np.nanstd(r2_scores):.4f}")
print(f"Correlação de Pearson média: {np.nanmean(pearson_scores):.4f} +/- {np.nanstd(pearson_scores):.4f}")

# Avaliação geral no dataset completo com o melhor modelo.
# Esta avaliação é para referência, as métricas da validação cruzada são mais robustas.
print("\n--- Métricas no Dataset Completo (com o melhor modelo treinado uma vez no dataset inteiro) ---")
final_model_for_full_data = grid_search.best_estimator_  # O modelo com os melhores hiperparâmetros.
final_model_for_full_data.fit(X, y)  # Treina o modelo no dataset completo.
final_predictions = final_model_for_full_data.predict(X)  # Faz previsões no dataset completo.

# Calcula as métricas globais.
global_mse = mean_squared_error(y, final_predictions)
global_mae = mean_absolute_error(y, final_predictions)
global_r2 = r2_score(y, final_predictions)
try:
    global_pearson, _ = pearsonr(y, final_predictions)
except ValueError:
    global_pearson = np.nan

print(f"MSE Global: {global_mse:.4f}")
print(f"MAE Global: {global_mae:.4f}")
print(f"R2 Global: {global_r2:.4f}")
print(f"Pearson Global: {global_pearson:.4f}")

print("\nFIM.")
