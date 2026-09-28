# Entraînement NVIDIA sous Windows / WSL2

Le training GPU utilise la distribution WSL2 `NNriot-GPU` et son environnement Python `/opt/nnriot-venv`. Le code et les données restent dans ce dépôt. Donner aux scripts des chemins relatifs au dépôt ou au format Linux.

TensorFlow récent ne prend plus en charge CUDA sous Windows natif. `.venv312/Scripts/python.exe` reste un environnement CPU. La procédure officielle est [TensorFlow avec CUDA sous WSL2](https://www.tensorflow.org/install/pip#windows-wsl2). `requirements-gpu.txt` installe TensorFlow 2.21.0 et ses bibliothèques CUDA.

## Vérifier le GPU

Depuis PowerShell, à la racine du dépôt :

```powershell
.\scripts\train_gpu.ps1 -Check
```

La commande affiche le GPU et effectue une multiplication matricielle sur `/GPU:0`. Elle échoue si CUDA est absent, si aucun GPU n'est visible ou si le calcul ne s'exécute pas sur GPU. Le lanceur applique cette vérification avant chaque entraînement dans le même processus TensorFlow, et active l'allocation progressive de la mémoire GPU.

## Entraîner DeepSets v3

```powershell
.\scripts\train_gpu.ps1 -TrainingArgs @(
  '--output-dir', 'runs/candidate_deepsets_v3_gpu',
  '--epochs', '5',
  '--batch-size', '64'
)
```

Le script conserve ses paramètres habituels, notamment le cache des partitions de 45 000 matchs. Choisir un nouveau `--output-dir` pour chaque essai : le script v3 remplace les fichiers déjà présents dans son répertoire de sortie. Un cache existant prend priorité sur `--samples` ; utiliser un autre `--splits-cache` pour changer la cohorte.

Pour le parcours pré-match avec base explicite :

```powershell
.\scripts\train_gpu.ps1 -Script scripts/train_pre_match_model.py -TrainingArgs @(
  '--db', 'runs/db_merge_20260924/consolidated.db',
  '--samples', '5000', '--epochs', '5', '--batch-size', '64',
  '--output', 'runs/pre_match_gpu/model.keras'
)
```

L'entraînement neuronal utilise le GPU ; la lecture SQLite, l'extraction NumPy des features et une partie de l'orchestration PCGrad restent sur CPU. Le GPU ne garantit donc pas une accélération pour tous les modèles ou petits batches. Ce lanceur ne convertit pas les modèles scikit-learn en modèles GPU.

## Installation sur une autre machine

Le pilote NVIDIA Windows doit être installé et `nvidia-smi` doit fonctionner. Créer une distribution WSL2 dédiée (nom et répertoire encore inutilisés) :

```powershell
wsl --install Ubuntu-24.04 --name NNriot-GPU --location "$PWD\.venv-wsl-system" --no-launch --web-download
wsl -d NNriot-GPU -u root --cd "$PWD" --exec bash scripts/setup_gpu_wsl.sh
.\scripts\train_gpu.ps1 -Check
```

L'installation télécharge plusieurs Go. Le disque de la distribution est stocké dans `.venv-wsl-system/`, déjà ignoré par Git, et contient l'environnement Python Linux. Ce dossier est le disque complet de la distribution WSL, pas un venv jetable. Le lanceur utilise `root` dans cette distribution dédiée ; cela n'élève pas PowerShell sous Windows.

Pour une distribution ou un environnement Python différents :

```powershell
.\scripts\train_gpu.ps1 -Distribution Ubuntu-24.04 -LinuxPython /opt/nnriot-venv/bin/python -Check
```

Le setup s'exécute dans la distribution choisie, installe les paquets Ubuntu Python, crée le venv, installe les dépendances, puis vérifie CUDA. Ne pas installer de pilote NVIDIA Linux dans WSL : le GPU est fourni par le pilote Windows.
