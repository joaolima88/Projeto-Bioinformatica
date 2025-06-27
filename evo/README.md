# Estrutura desta pasta:

evo/
├── preprocessamento_e_filtragem.py             # Script para limpeza, filtragem e escalagem dos dados brutos
├── preprocessamento_e_filtragem.ipynb          # Script para limpeza, filtragem e escalagem dos dados brutos em formato ipynb (possui informação adicional sobre o processamento)
├── evo_training.py                             # Script para treinar e avaliar o modelo SVM usando embeddings EVO
├── README.md                                   # Este ficheiro: instruções e visão geral do projeto
│
├── data/                                       # Pasta para dados
│   ├── promoters_RBS.xlsx                      # Dataset original de constructs (promotores + RBS + medições)
│   ├── promoters.xlsx                          # Dataset de sequências de promotores
│   ├── RBS.xlsx                                # Dataset de sequências de RBS
│   ├── training_data.csv                       # Dataset final processado
│   └── training_data.xlsx                      # Versão Excel do dataset processado (para visualização)
│
├── embeddings/                                 # Pasta para guardar os embeddings de sequência gerados
│   └── evo_embeddings.npy                      # Embeddings de sequências gerados pelo EVO
│
├── scalers/                                    # Pasta para guardar os objetos scaler
│   ├── scaler_standard.pkl                     # Objeto StandardScaler
│   └── scaler_robust.pkl                       # Objeto RobustScaler
│
└── results/                                    # Pasta para guardar resultados dos modelos treinados, logs e gráficos

---

# Instruções de Setup do Ambiente para o Projeto EVO


## 1. Requisitos de Hardware (GPU)

O modelo EVO utiliza internamente o FlashAttention-2 para operações otimizadas. O FlashAttention-2 tem requisitos de hardware específicos para funcionar com aceleração GPU:

- GPU NVIDIA com arquitetura Ampere (Compute Capability 8.0/sm_80) ou superior.
  - Exemplos compatíveis: NVIDIA A100, RTX 30 Series (ex: RTX 3060, 3070, 3080, 3090), RTX 40 Series (ex: RTX 4070, 4080, 4090), H100.
  - GPUs mais antigas (ex: Turing, Pascal, Volta) não são suportadas e causarão erros.
- Sistema Operacional: Linux recomendado. Suporte experimental para Windows a partir da v2.3.2 (requer testes adicionais).
- VRAM: Para o modelo `evo-1-8k-base` (6.45B parâmetros), são necessários 13-16GB de VRAM (half precision). Erros `CUDA out of memory` indicam falta de VRAM.


---


## 2. Setup do Ambiente

### 2.1. Instalar o Conda/Miniconda/Mamba

Se ainda não tiver, faça o download e instale a versão mais recente para o seu sistema operativo a partir dos respetivos websites.

Após a instalação, certifique-se de que o Conda/Mamba está inicializado no seu terminal.

---

### 2.2. Criar e Ativar o Ambiente Dedicado (Usando Ficheiro YAML)

Para garantir um ambiente consistente com as dependências básicas, utilize o ficheiro environment.yml:

name: evo-design
channels:
  - bioconda
  - conda-forge
  - defaults
dependencies:
  - bioconda::prodigal
  - python=3.10
  - biopython



Passos para criar o ambiente:

1. Navegue para a diretoria onde guardou o environment.yml
2. crie o ambiente com `conda env create -f environment.yml`
3. Ative o ambiente
   
---

### 2.3. Configurar o CUDA Toolkit (Essencial para GPU)

Este passo é crítico para que as bibliotecas baseadas em CUDA (como PyTorch e FlashAttention) compilem e funcionem corretamente.

Verificar instalação do CUDA Toolkit:

1. Verifique se o nvcc está no PATH e se CUDA_HOME está definido:

	`which nvcc`
	`echo $CUDA_HOME`
   
2. Se não estiverem, tente localizar a instalação do CUDA Toolkit:

	`ls -d /usr/local/cuda`
	`ls -d /opt/cuda`
   
3. Após encontrar o caminho (ex: /usr/local/cuda-12.2), defina as variáveis de ambiente manualmente (substituindo X.Y pela sua versão CUDA):

	`export CUDA_HOME=/path/to/cuda-X.Y`
	`export PATH=$CUDA_HOME/bin:$PATH`
	`export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`
   
---

### 2.4. Instalar Bibliotecas Python Principais

Importante: A ordem de instalação é crucial para evitar conflitos e garantir a compilação correta dos kernels.


#### a) Instalar PyTorch com suporte CUDA (PRIMEIRO E ESSENCIAL)

É essencial instalar o PyTorch com suporte a CUDA no ambiente.

Consulte o site oficial do PyTorch (https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) para a versão mais recente e o comando conda install correto que corresponda à sua versão de CUDA Toolkit e arquitetura de GPU.

Exemplo para CUDA 12.1: 
 
 `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y` # Verifique a sua versão de CUDA e ajuste o comando conforme necessário.)


#### b) Instalar as restantes bibliotecas via pip: 

- Ninja (aconselhado para compilação otimizada mas não é essencial):

	`pip install ninja`

  - Verifique se o Ninja está funcional:
 
	`ninja --version` e depois  `echo $?`  # Deve retornar 0
 
  - Se falhar, reinstale:

	`pip uninstall -y ninja && pip install ninja`
 
  - Sem Ninja, a compilação pode levar até 2h (vs. 3-5 minutos com Ninja em máquinas multicore).


- Stripedhyena

	`pip install stripedhyena==0.2.2 evo-model scikit-learn tqdm tokenizers transformers pandas`


O stripedhyena deve puxar o flashattention automaticamente, mas caso não aconteça:

 - FlashAttention-2 (sem isolamento de build):

	`pip install flash-attn --no-build-isolation`



---


## Repositórios a consultar:

FlashAttention - https://github.com/Dao-AILab/flash-attention
StripedHyena - https://github.com/togethercomputer/stripedhyena
evo - https://github.com/evo-design/evo?tab=readme-ov-file


