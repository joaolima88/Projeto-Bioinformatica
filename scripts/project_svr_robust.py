# -*- coding: utf-8 -*-
"""Project_SVR_robust.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sPyWS0VA_VSYDvZJzLJ7ldZW9rY6ZSmj
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import os
import sys
import datetime
import joblib

from google.colab import drive
drive.mount('/content/drive')

BASE_DRIVE_PATH = "/content/drive/MyDrive"

TRAINING_DATA_FILE = os.path.join(BASE_DRIVE_PATH, "training_data.csv")

DNABERT_EMBEDDINGS_FILE = os.path.join(BASE_DRIVE_PATH, "dnabert_embeddings.npy")

NORMALIZATION_SCALER_FILE = os.path.join(BASE_DRIVE_PATH, "scaler_robust.pkl")

RESULTS_OUTPUT_DIR = BASE_DRIVE_PATH
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
RESULTS_FILE_PREFIX = "svm_robust_optimization_results"
PREDICTIONS_DATA_PREFIX = "svm_robust_predictions_data"

# 2. Carregar Dados Originais e Embeddings Pré-Calculados
try:
    df = pd.read_csv(TRAINING_DATA_FILE)
except FileNotFoundError:
    print(f"ERRO: O ficheiro de dados original '{TRAINING_DATA_FILE}' não foi encontrado.")
    sys.exit(1)

if not all(col in df.columns for col in ['target', 'sequence', 'prot_scaled']):
    print("ERRO: Colunas inesperadas no DataFrame original")
    sys.exit(1)

y_scaled = df['prot_scaled'].values.ravel()


scaler = None
try:
    scaler = joblib.load(NORMALIZATION_SCALER_FILE)
except FileNotFoundError:
    print(f"ERRO: O ficheiro RobustScaler '{NORMALIZATION_SCALER_FILE}' não foi encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO: Falha ao carregar RobustScaler de {NORMALIZATION_SCALER_FILE}: {e}")
    sys.exit(1)

true_original_target_values = scaler.inverse_transform(df['prot_scaled'].values.reshape(-1, 1)).flatten()

try:
    X = np.load(DNABERT_EMBEDDINGS_FILE)
    if X.shape[0] != len(df):
        print(f"AVISO: O número de amostras nos embeddings ({X.shape[0]}) não corresponde ao número de amostras no CSV ({len(df)}).")
except FileNotFoundError:
    print(f"ERRO: O ficheiro de embeddings '{DNABERT_EMBEDDINGS_FILE}' não foi encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO: Falha ao carregar o ficheiro de embeddings '{DNABERT_EMBEDDINGS_FILE}': {e}")
    sys.exit(1)

from sklearn.experimental import enable_halving_search_cv  # Necessário para ativar HalvingRandomSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

# --- 3. Treinar o SVM (SVR - Support Vector Regressor) com HalvingRandomSearchCV ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

param_distributions = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

svr = SVR()

random_search = HalvingRandomSearchCV(
    estimator=SVR(),
    param_distributions=param_distributions,
    factor=3,
    resource='n_samples',
    min_resources=500,
    max_resources=9000,
    n_candidates=50,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X, y_scaled)

print(f"\nMelhores hiperparâmetros encontrados: {random_search.best_params_}")
best_svr_model = random_search.best_estimator_

# --- 4. Avaliar o Modelo ---
print("\n--- Avaliação com validação cruzada nos melhores parâmetros ---")
mse_scores_scaled = []
mae_scores_scaled = []
r2_scores_scaled = []
pearson_scores_scaled = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    best_svr_model.fit(X_train, y_train)
    y_pred_scaled = best_svr_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_scaled)
    mae = mean_absolute_error(y_test, y_pred_scaled)
    r2 = r2_score(y_test, y_pred_scaled)

    try:
        pearson_corr, _ = pearsonr(y_test, y_pred_scaled)
    except ValueError:
        pearson_corr = np.nan

    mse_scores_scaled.append(mse)
    mae_scores_scaled.append(mae)
    r2_scores_scaled.append(r2)
    pearson_scores_scaled.append(pearson_corr)

    print(f"Fold {fold+1}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Pearson={pearson_corr:.4f}")

print("\n--- Resultados Finais da Validação Cruzada ---")
final_cv_results = {
    "MSE médio (escala escalada)": f"{np.nanmean(mse_scores_scaled):.4f} +/- {np.nanstd(mse_scores_scaled):.4f}",
    "MAE médio (escala escalada)": f"{np.nanmean(mae_scores_scaled):.4f} +/- {np.nanstd(mae_scores_scaled):.4f}",
    "R2 médio (escala escalada)": f"{np.nanmean(r2_scores_scaled):.4f} +/- {np.nanstd(r2_scores_scaled):.4f}",
    "Correlação de Pearson média (escala escalada)": f"{np.nanmean(pearson_scores_scaled):.4f} +/- {np.nanstd(pearson_scores_scaled):.4f}"
}
for metric, value in final_cv_results.items():
    print(f"{metric}: {value}")

# --- 5. Avaliação Final no Dataset Completo e Guardar Dados para Gráficos ---
print("\n--- Métricas no Dataset Completo (com o melhor modelo treinado uma vez no dataset inteiro) ---")
final_model_for_full_data = random_search.best_estimator_
final_model_for_full_data.fit(X, y_scaled)
final_predictions_standardized = final_model_for_full_data.predict(X)

final_predictions_original = scaler.inverse_transform(final_predictions_standardized.reshape(-1, 1)).flatten()

global_mse = mean_squared_error(true_original_target_values, final_predictions_original)
global_mae = mean_absolute_error(true_original_target_values, final_predictions_original)
global_r2 = r2_score(true_original_target_values, final_predictions_original)
try:
    global_pearson, _ = pearsonr(true_original_target_values.flatten(), final_predictions_original.flatten())
except ValueError:
    global_pearson = np.nan

global_rmse = np.sqrt(global_mse)


mean_of_original_targets = np.mean(true_original_target_values)

if mean_of_original_targets != 0:
    pmae = (global_mae / mean_of_original_targets) * 100
    prmse = (global_rmse / mean_of_original_targets) * 100
else:
    pmae = np.nan
    prmse = np.nan

global_results = {
    "MSE Global (escala original)": f"{global_mse:.4f}",
    "MAE Global (escala original)": f"{global_mae:.4f}",
    "MAE Percentual (vs. Média)": f"{pmae:.2f}%",
    "RMSE Global (escala original)": f"{global_rmse:.4f}",
    "RMSE Percentual (vs. Média)": f"{prmse:.2f}%",
    "R2 Global (escala original)": f"{global_r2:.4f}",
    "Pearson Global (escala original)": f"{global_pearson:.4f}"
}
for metric, value in global_results.items():
    print(f"{metric}: {value}")


min_original_prot = np.min(true_original_target_values)
max_original_prot = np.max(true_original_target_values)
std_original_prot = np.std(true_original_target_values)
amplitude_original_prot = max_original_prot - min_original_prot

print(f"\n--- Escala dos Valores Originais de Expressão Proteica ---")
print(f"  Mínimo: {min_original_prot:.4f}")
print(f"  Máximo: {max_original_prot:.4f}")
print(f"  Média: {mean_of_original_targets:.4f}")
print(f"  Desvio Padrão: {std_original_prot:.4f}")
print(f"  Amplitude: {amplitude_original_prot:.4f}")

# --- 6. Guardar Resultados e Dados para Gráficos ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = os.path.join(RESULTS_OUTPUT_DIR, f"{RESULTS_FILE_PREFIX}_{timestamp}.txt")
predictions_data_filename = os.path.join(RESULTS_OUTPUT_DIR, f"{PREDICTIONS_DATA_PREFIX}_{timestamp}.npy")
model_save_filename = os.path.join(RESULTS_OUTPUT_DIR, f"best_svm_model_robust_{timestamp}.joblib")


print(f"\n--- A guardar resultados em: {results_filename} ---")
with open(results_filename, 'w') as f:
    f.write(f"Resultados da Otimização do SVM com RobustScaler\n")
    f.write(f"Data e Hora: {timestamp}\n")
    f.write(f"----------------------------------------------------\n\n")

    f.write(f"Melhores Hiperparâmetros Encontrados:\n")
    for param, value in random_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\n")

    f.write(f"Resultados da Validação Cruzada (Médias e Desvios Padrão - ESCALA ESCALADA (RobustScaler)):\n")
    for metric, value in final_cv_results.items():
        f.write(f"  {metric}: {value}\n")

    f.write(f"Métricas no Dataset Completo (Com o melhor modelo treinado uma vez no dataset inteiro - ESCALA ORIGINAL):\n")
    for metric, value in global_results.items():
        f.write(f"  {metric}: {value}\n")
    f.write(f"\n")
    f.write(f"--- Escala dos Valores Originais de Expressão Proteica ---\n")
    f.write(f"  Mínimo: {min_original_prot:.4f}\n")
    f.write(f"  Máximo: {max_original_prot:.4f}\n")
    f.write(f"  Média: {mean_of_original_targets:.4f}\n")
    f.write(f"  Desvio Padrão: {std_original_prot:.4f}\n")
    f.write(f"  Amplitude: {amplitude_original_prot:.4f}\n")
    f.write(f"\n")

    f.write(f"Informações Adicionais:\n")
    f.write(f"  Número de Amostras: {X.shape[0]}\n")
    f.write(f"  Número de Features: {X.shape[1]}\n")
    f.write(f"  Tipo de Embeddings: DNABERT\n")
    f.write(f"  Folds de Validação Cruzada: {n_splits}\n")
    f.write(f"  Iterações de RandomizedSearchCV: {random_search.n_iterations_}\n")
    f.write(f"  Ficheiro de Embeddings Usado: {os.path.basename(DNABERT_EMBEDDINGS_FILE)}\n")
    f.write(f"  Ficheiro de Modelo SVM Guardado: {os.path.basename(model_save_filename)}\n")
    f.write(f"  Ficheiro de Dados de Previsões Guardado: {os.path.basename(predictions_data_filename)}\n")
    f.write(f"  Ficheiro do Scaler Guardado: {os.path.basename(NORMALIZATION_SCALER_FILE)}\n") # ALTERADO


print(f"Resultados guardados com sucesso em {results_filename}.")

data_for_plotting = np.vstack((true_original_target_values.flatten(), final_predictions_original)).T # ALTERADO
np.save(predictions_data_filename, data_for_plotting)
print(f"Valores reais (escala original) e previstos (escala original) guardados em {predictions_data_filename} para análise de gráficos.")

joblib.dump(best_svr_model, model_save_filename)
print(f"Melhor modelo SVM guardado em: {model_save_filename}")

print("\nScript de otimização de SVM concluído.")