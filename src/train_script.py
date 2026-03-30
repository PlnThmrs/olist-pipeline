# src/train_script.py (modifié)
import argparse
import os
import logging
import joblib
import shutil
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Optional AzureML Run import (wrapped)
try:
    from azureml.core import Run
    _HAS_AZUREML = True
except Exception:
    _HAS_AZUREML = False

parser = argparse.ArgumentParser()
parser.add_argument('--train_data',    type=str,   required=True)
parser.add_argument('--model_type',    type=str,   default='random_forest')
parser.add_argument('--n_estimators',  type=int,   default=100)
parser.add_argument('--max_depth',     type=int,   default=10)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--model_output',  type=str,   required=True)
args = parser.parse_args()

# Chargement
X_train = pd.read_csv(f'{args.train_data}/X_train.csv')
y_train = pd.read_csv(f'{args.train_data}/y_train.csv').squeeze()
log.info(f'X_train : {X_train.shape}')

# Selection du modele
models = {
    'random_forest': RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42, n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42
    ),
    'ridge': Ridge(alpha=1.0),
}
try:
    import lightgbm as lgb
    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42, n_jobs=-1, verbose=-1
    )
except ImportError:
    log.warning('lightgbm non disponible')

if args.model_type not in models:
    raise ValueError(f'model_type inconnu: {args.model_type}')
model = models[args.model_type]

# Entrainement + MLflow
with mlflow.start_run():
    mlflow.log_params({
        'model_type':    args.model_type,
        'n_estimators':  args.n_estimators,
        'max_depth':     args.max_depth,
        'learning_rate': args.learning_rate,
        'n_train':       len(X_train),
    })

    log.info(f'Entrainement {args.model_type}...')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    train_r2  = float(model.score(X_train, y_train))
    mlflow.log_metrics({'train_mae': train_mae, 'train_r2': train_r2})
    log.info(f'Train MAE={train_mae:.2f} | R2={train_r2:.4f}')

    # Préparer dossier de sortie
    os.makedirs(args.model_output, exist_ok=True)

    # --- Option B : sauvegarde locale MLflow puis upload artefacts ---
    local_model_dir = os.path.join(args.model_output, "model_local")
    os.makedirs(local_model_dir, exist_ok=True)

    # 1) Sauvegarder le modèle au format MLflow localement
    #    (évite d'appeler l'endpoint logged-models côté serveur)
    mlflow.sklearn.save_model(sk_model=model, path=local_model_dir)
    log.info(f"Model saved locally to {local_model_dir}")

    # 2) Upload des artefacts : si on est dans AzureML, utiliser Run.upload_folder
    uploaded_via_azureml = False
    if _HAS_AZUREML:
        try:
            run = Run.get_context()
            # upload_folder met le contenu dans les artefacts de l'exécution AzureML
            run.upload_folder(name="model", path=local_model_dir)
            log.info("Model artifacts uploaded to AzureML run via Run.upload_folder(name='model').")
            uploaded_via_azureml = True
        except Exception as e:
            log.warning(f"Upload via AzureML Run failed: {e}. Falling back to mlflow.log_artifacts.")

    # 3) Fallback : si pas d'AzureML ou si upload a échoué, utiliser mlflow.log_artifacts
    if not uploaded_via_azureml:
        try:
            mlflow.log_artifacts(local_model_dir, artifact_path="model")
            log.info("Model artifacts logged to MLflow via mlflow.log_artifacts(local_model_dir, artifact_path='model').")
        except Exception as e:
            log.error(f"mlflow.log_artifacts failed: {e}. Artefacts locaux conservés dans {local_model_dir}.")

    # 4) Sauvegarde locale supplémentaire (pickle) pour un chargement simple côté scoring
    try:
        joblib.dump(model, f'{args.model_output}/model.pkl')
        log.info(f"Model pickle saved to {args.model_output}/model.pkl")
    except Exception as e:
        log.warning(f"joblib.dump failed: {e}")

    # Copier fichiers utilitaires si présents
    for fname in ['scaler.pkl', 'feature_names.txt']:
        src = f'{args.train_data}/{fname}'
        if os.path.exists(src):
            shutil.copy(src, f'{args.model_output}/{fname}')

log.info(f'Modele sauvegarde -> {args.model_output}')
