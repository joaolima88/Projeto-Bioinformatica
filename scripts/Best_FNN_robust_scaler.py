#%%
# --- 0. Importações de Bibliotecas Essenciais ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
import os
import sys
import datetime
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt # NOVO: Importar matplotlib para plotagem
#%%
# --- 1. Configurações de Ficheiros ---

BASE_PATH = "." # O diretório atual onde este script está a ser executado

TRAINING_DATA_FILE = os.path.join(BASE_PATH, "training_data.csv")
DNABERT_EMBEDDINGS_FILE = os.path.join(BASE_PATH, "dnabert_embeddings.npy")
NORMALIZATION_SCALER_FILE = os.path.join(BASE_PATH, "scaler_robust.pkl")

RESULTS_OUTPUT_DIR = BASE_PATH
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

RESULTS_FILE_PREFIX = "fnn_optimization_results"
PREDICTIONS_DATA_PREFIX = "fnn_predictions_data"
#%%
# --- 2. Configurações da FNN e Novas Configurações de Treino ---
FNN_HIDDEN_DIM_1 = 256
FNN_HIDDEN_DIM_2 = 128
FNN_LEARNING_RATE = 0.001
FNN_NUM_EPOCHS = 250
FNN_BATCH_SIZE = 64

FNN_DROPOUT_RATE = 0.2
FNN_WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 25

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_MIN_LR = 1e-6
#%%
# --- FUNÇÃO AUXILIAR ---
def plot_loss_curves(train_losses, val_losses, fold_num=None, title_suffix="", save_dir=None, prefix="loss_curve"):
    """
    Cria as curvas de perda de treino e validação.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Perda de Treino')

    if val_losses:
        plt.plot(val_losses, label='Perda de Validação')

    plt.xlabel('Época')
    plt.ylabel('Perda (MSE)')

    if fold_num is not None:
        plt.title(f'Curvas de Perda da FNN - Fold {fold_num} {title_suffix}')
    else:
        plt.title(f'Curvas de Perda da FNN - {title_suffix}')

    plt.legend()
    plt.grid(True)

    if save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if fold_num is not None:
            filename = os.path.join(save_dir, f"{prefix}_fold_{fold_num}_{timestamp}.png")
        else:
            filename = os.path.join(save_dir, f"{prefix}_{timestamp}.png")
        plt.savefig(filename)
        print(f"Curva de perda salva em: {filename}")
    plt.close()
#%%
# --- 3. Carregar Dados Originais e Embeddings Pré-Calculados ---
print("\n--- A carregar dados e embeddings pré-calculados ---")

try:
    df = pd.read_csv(TRAINING_DATA_FILE)
    print(f"Dados originais carregados com sucesso de {TRAINING_DATA_FILE}. Dimensão: {df.shape}")
except FileNotFoundError:
    print(f"ERRO: O ficheiro de dados original '{TRAINING_DATA_FILE}' não foi encontrado.")
    sys.exit(1)

if not all(col in df.columns for col in ['target', 'sequence', 'prot_scaled']):
    print("ERRO: Colunas inesperadas no DataFrame original.")
    sys.exit(1)

y_scaled = torch.tensor(df['prot_scaled'].values, dtype=torch.float32).unsqueeze(1)

scaler = None
try:
    scaler = joblib.load(NORMALIZATION_SCALER_FILE)
    print(f"RobustScaler carregado com sucesso de: {NORMALIZATION_SCALER_FILE}")
    print(f"Centro do scaler para desescalar: {scaler.center_[0]:.4f}, Escala: {scaler.scale_[0]:.4f}")
except FileNotFoundError:
    print(f"ERRO: O ficheiro RobustScaler '{NORMALIZATION_SCALER_FILE}' não foi encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO: Falha ao carregar RobustScaler de {NORMALIZATION_SCALER_FILE}: {e}")
    sys.exit(1)

true_original_target_values = scaler.inverse_transform(df['prot_scaled'].values.reshape(-1, 1)).flatten()

try:
    X_embeddings = np.load(DNABERT_EMBEDDINGS_FILE)
    X_embeddings = torch.tensor(X_embeddings, dtype=torch.float32)
    if X_embeddings.shape[0] != len(df):
        print(f"AVISO: O número de amostras nos embeddings ({X_embeddings.shape[0]}) não corresponde ao número de amostras no CSV ({len(df)}).")
        sys.exit(1)
except FileNotFoundError:
    print(f"ERRO: O ficheiro de embeddings '{DNABERT_EMBEDDINGS_FILE}' não foi encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO: Falha ao carregar o ficheiro de embeddings '{DNABERT_EMBEDDINGS_FILE}': {e}")
    sys.exit(1)

print(f"\nDimensão de X (features): {X_embeddings.shape}")
print(f"Dimensão de y (target 'prot_scaled' JÁ ESCALADO): {y_scaled.shape}")
#%%
# --- 4. Definição da Feedforward Neural Network (FNN) ---
class FNN(nn.Module):

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim=1, dropout_rate=0.0):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate) # Dropout após a primeira camada

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2) # Segunda camada oculta
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate) # Dropout após a segunda camada

        self.fc3 = nn.Linear(hidden_dim_2, output_dim) # Camada de saída

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x) # Passagem pela segunda camada
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"dispositivo: {device}")
#%%
# --- 5. Treinar e Avaliar a FNN com Validação Cruzada ---

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scores_scaled = []
mae_scores_scaled = []
r2_scores_scaled = []
pearson_scores_scaled = []

for fold, (train_index, test_index) in enumerate(kf.split(X_embeddings)):
    print(f"\n--- FNN - Fold {fold+1}/{n_splits} ---")

    X_train, X_test = X_embeddings[train_index], X_embeddings[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=FNN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FNN_BATCH_SIZE, shuffle=False)

    input_dim = X_embeddings.shape[1]
    model = FNN(input_dim, FNN_HIDDEN_DIM_1, FNN_HIDDEN_DIM_2, dropout_rate=FNN_DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=FNN_LEARNING_RATE, weight_decay=FNN_WEIGHT_DECAY)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch_model_state = None

    train_losses_fold = []
    val_losses_fold = []

    for epoch in range(FNN_NUM_EPOCHS):

        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Avaliação no conjunto de teste (para Early Stopping e Learning Rate Scheduler)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch_val, y_batch_val in test_loader:
                X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                outputs_val = model(X_batch_val)
                loss_val = criterion(outputs_val, y_batch_val)
                total_val_loss += loss_val.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)

        train_losses_fold.append(avg_train_loss)
        val_losses_fold.append(avg_val_loss)

        # Step do scheduler com a perda de validação
        scheduler.step(avg_val_loss)

        # Lógica de Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  Early Stopping at Epoch {epoch+1} for Fold {fold+1}. Best Val Loss: {best_val_loss:.4f}")
                model.load_state_dict(best_epoch_model_state)
                break


        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == FNN_NUM_EPOCHS - 1 or epochs_no_improve == EARLY_STOPPING_PATIENCE:
            print(f'  Epoch [{epoch+1}/{FNN_NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Current LR: {current_lr:.6f}')

    plot_loss_curves(train_losses_fold, val_losses_fold, fold_num=fold+1, save_dir=RESULTS_OUTPUT_DIR, prefix="fnn_loss_curve_deeper_lr_sched")


    model.eval()
    fold_predictions_scaled = []
    fold_actual_scaled = []
    with torch.no_grad():
        for X_batch_test_eval, y_batch_test_eval in test_loader:
            X_batch_test_eval, y_batch_test_eval = X_batch_test_eval.to(device), y_batch_test_eval.to(device)
            outputs_test_eval = model(X_batch_test_eval)
            fold_predictions_scaled.extend(outputs_test_eval.cpu().numpy().flatten())
            fold_actual_scaled.extend(y_batch_test_eval.cpu().numpy().flatten())

    fold_predictions_scaled = np.array(fold_predictions_scaled)
    fold_actual_scaled = np.array(fold_actual_scaled)

    mse = mean_squared_error(fold_actual_scaled, fold_predictions_scaled)
    mae = mean_absolute_error(fold_actual_scaled, fold_predictions_scaled)
    r2 = r2_score(fold_actual_scaled, fold_predictions_scaled)

    try:
        pearson_corr, _ = pearsonr(fold_actual_scaled, fold_predictions_scaled)
    except ValueError:
        pearson_corr = np.nan

    mse_scores_scaled.append(mse)
    mae_scores_scaled.append(mae)
    r2_scores_scaled.append(r2)
    pearson_scores_scaled.append(pearson_corr)

    print(f"Fold {fold+1} Resultados (escala escalada): MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Pearson={pearson_corr:.4f}")


print("\n--- Resultados Finais da Validação Cruzada (Médias e Desvios Padrão) ---")
final_cv_results = {
    "MSE médio (escala escalada)": f"{np.nanmean(mse_scores_scaled):.4f} +/- {np.nanstd(mse_scores_scaled):.4f}",
    "MAE médio (escala escalada)": f"{np.nanmean(mae_scores_scaled):.4f} +/- {np.nanstd(mae_scores_scaled):.4f}",
    "R2 médio (escala escalada)": f"{np.nanmean(r2_scores_scaled):.4f} +/- {np.nanstd(r2_scores_scaled):.4f}",
    "Correlação de Pearson média (escala escalada)": f"{np.nanmean(pearson_scores_scaled):.4f} +/- {np.nanstd(pearson_scores_scaled):.4f}"
}
for metric, value in final_cv_results.items():
    print(f"{metric}: {value}")
#%%
# --- 6. Avaliação Final no Dataset Completo ---
print("\n--- Métricas no Dataset Completo (com o modelo final treinado no dataset inteiro) ---")

input_dim = X_embeddings.shape[1]
# ALTERADO: Passar ambas as dimensões ocultas para o construtor da FNN para o modelo final
final_model = FNN(input_dim, FNN_HIDDEN_DIM_1, FNN_HIDDEN_DIM_2, dropout_rate=FNN_DROPOUT_RATE).to(device)
final_criterion = nn.MSELoss()
# ALTERADO: Adicionado weight_decay ao otimizador do modelo final
final_optimizer = optim.Adam(final_model.parameters(), lr=FNN_LEARNING_RATE, weight_decay=FNN_WEIGHT_DECAY)

# NOVO: Agendador de taxa de aprendizagem para o modelo final
final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    final_optimizer,
    mode='min',
    factor=LR_SCHEDULER_FACTOR,
    patience=LR_SCHEDULER_PATIENCE,
    min_lr=LR_SCHEDULER_MIN_LR,
)

full_dataset = TensorDataset(X_embeddings, y_scaled)
full_loader = DataLoader(full_dataset, batch_size=FNN_BATCH_SIZE, shuffle=True)

# Treino do modelo final no dataset completo (com Early Stopping)
best_val_loss_final_model = float('inf') # Reutiliza para monitorizar a perda no treino final
epochs_no_improve_final_model = 0
best_final_model_state = None

train_losses_final_model = []

for epoch in range(FNN_NUM_EPOCHS):
    final_model.train()
    total_train_loss_final_model = 0
    for X_batch, y_batch in full_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        final_optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = final_criterion(outputs, y_batch)
        loss.backward()
        final_optimizer.step()
        total_train_loss_final_model += loss.item()

    avg_train_loss_final_model = total_train_loss_final_model / len(full_loader)
    train_losses_final_model.append(avg_train_loss_final_model)

    # Step do scheduler com a perda de treino (já que é o conjunto completo)
    final_scheduler.step(avg_train_loss_final_model)

    # Lógica de Early Stopping para o modelo final
    if avg_train_loss_final_model < best_val_loss_final_model: # Monitoriza a perda de treino para ES
        best_val_loss_final_model = avg_train_loss_final_model
        epochs_no_improve_final_model = 0
        best_final_model_state = final_model.state_dict()
    else:
        epochs_no_improve_final_model += 1
        if epochs_no_improve_final_model >= EARLY_STOPPING_PATIENCE:
            print(f"  Early Stopping for Final Model at Epoch {epoch+1}. Best Train Loss: {best_val_loss_final_model:.4f}")
            final_model.load_state_dict(best_final_model_state)
            break

    current_lr_final_model = final_optimizer.param_groups[0]['lr']
    if (epoch + 1) % 20 == 0 or epoch == FNN_NUM_EPOCHS - 1 or epochs_no_improve_final_model == EARLY_STOPPING_PATIENCE:
        print(f'  Final Model Epoch [{epoch+1}/{FNN_NUM_EPOCHS}], Train Loss: {avg_train_loss_final_model:.4f}, Current LR: {current_lr_final_model:.6f}')

plot_loss_curves(train_losses_final_model, [], fold_num=None, title_suffix="Modelo Final (Treino Completo)", save_dir=RESULTS_OUTPUT_DIR, prefix="fnn_final_model_loss_curve_deeper_lr_sched")

# Fazer previsões no dataset completo (na escala escalada)
final_model.eval()
final_predictions_scaled = []
with torch.no_grad():
    for X_batch_test, _ in DataLoader(TensorDataset(X_embeddings, y_scaled), batch_size=FNN_BATCH_SIZE, shuffle=False):
        X_batch_test = X_batch_test.to(device)
        outputs_test = final_model(X_batch_test)
        final_predictions_scaled.extend(outputs_test.cpu().numpy().flatten())

final_predictions_scaled = np.array(final_predictions_scaled)

final_predictions_original = scaler.inverse_transform(final_predictions_scaled.reshape(-1, 1)).flatten()

global_mse = mean_squared_error(true_original_target_values, final_predictions_original)
global_mae = mean_absolute_error(true_original_target_values, final_predictions_original)
global_r2 = r2_score(true_original_target_values, final_predictions_original)
try:
    global_pearson, _ = pearsonr(true_original_target_values.flatten(), final_predictions_original.flatten())
except ValueError:
    global_pearson = np.nan

# --- Cálculo das Percentagens de Erro ---
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

# --- Escala dos Valores Originais de Expressão Proteica ---
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

#%%
# --- 7. Guardar Resultados e Modelo ---
import os

# Criar a pasta de resultados se não existir
results_folder = "results_FNN_robust"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"Pasta '{results_folder}' criada com sucesso.")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Definir os caminhos dos ficheiros na pasta de resultados
results_filename = os.path.join(results_folder, f"{RESULTS_FILE_PREFIX}_{timestamp}.txt")
predictions_data_filename = os.path.join(results_folder, f"{PREDICTIONS_DATA_PREFIX}_{timestamp}.npy")
model_save_filename = os.path.join(results_folder, f"best_fnn_model_robust_{timestamp}.pth")

print(f"\n--- A guardar resultados em: {results_filename} ---")

with open(results_filename, 'w') as f:
    f.write(f"Resultados da Otimização da FNN com RobustScaler (Melhorada)\n")
    f.write(f"Data e Hora: {timestamp}\n")
    f.write(f"----------------------------------------------------\n\n")
    f.write(f"Hiperparâmetros da FNN:\n")
    f.write(f"  Dimensão Oculta 1: {FNN_HIDDEN_DIM_1}\n")
    f.write(f"  Dimensão Oculta 2: {FNN_HIDDEN_DIM_2}\n")
    f.write(f"  Taxa de Aprendizagem: {FNN_LEARNING_RATE}\n")
    f.write(f"  Número de Épocas Máximo: {FNN_NUM_EPOCHS}\n")
    f.write(f"  Tamanho do Lote: {FNN_BATCH_SIZE}\n")
    f.write(f"  Taxa de Dropout: {FNN_DROPOUT_RATE}\n")
    f.write(f"  Weight Decay (L2 Reg.): {FNN_WEIGHT_DECAY}\n")
    f.write(f"  Patience para Early Stopping: {EARLY_STOPPING_PATIENCE}\n")
    f.write(f"\n")
    f.write(f"Resultados da Validação Cruzada (Médias e Desvios Padrão - ESCALA ESCALADA (RobustScaler)):\n")
    for metric, value in final_cv_results.items():
        f.write(f"  {metric}: {value}\n")
    f.write(f"\n")
    f.write(f"Métricas no Dataset Completo (Com o modelo final treinado uma vez no dataset inteiro - ESCALA ORIGINAL):\n")
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
    f.write(f"  Número de Amostras: {X_embeddings.shape[0]}\n")
    f.write(f"  Número de Features: {X_embeddings.shape[1]}\n")
    f.write(f"  Tipo de Embeddings: DNABERT\n")
    f.write(f"  Folds de Validação Cruzada: {n_splits}\n")
    f.write(f"  Ficheiro de Embeddings Usado: {os.path.basename(DNABERT_EMBEDDINGS_FILE)}\n")
    f.write(f"  Ficheiro de Modelo FNN Guardado: {os.path.basename(model_save_filename)}\n")
    f.write(f"  Ficheiro de Dados de Previsões Guardado: {os.path.basename(predictions_data_filename)}\n")
    f.write(f"  Ficheiro do Scaler Guardado: {os.path.basename(NORMALIZATION_SCALER_FILE)}\n")

print(f"Resultados guardados com sucesso em {results_filename}.")

# Guardar o modelo
torch.save(final_model.state_dict(), model_save_filename)
print(f"Melhor modelo FNN guardado em: {model_save_filename}.")

# Guardar os dados de previsões
data_for_plotting = np.vstack((true_original_target_values.flatten(), final_predictions_original)).T
np.save(predictions_data_filename, data_for_plotting)
print(f"Valores reais (escala original) e previstos (escala original) guardados em {predictions_data_filename} para análise de gráficos.")

print("\nScript de treino e avaliação da FNN concluído.")
print(f"Todos os ficheiros foram guardados na pasta: {results_folder}")