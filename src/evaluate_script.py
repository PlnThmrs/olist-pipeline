# src/evaluate_script.py (modifié)
import argparse
import os
import json
import logging
import joblib
import math

import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_input', type=str, required=True)
parser.add_argument('--test_data',   type=str, required=True)
parser.add_argument('--eval_output', type=str, required=True)
args = parser.parse_args()

# Chargement
model = joblib.load(f'{args.model_input}/model.pkl')
X_test = pd.read_csv(f'{args.test_data}/X_test.csv')
y_test = pd.read_csv(f'{args.test_data}/y_test.csv').squeeze()
log.info(f'Test : {X_test.shape} | Modele : {type(model).__name__}')

# Predictions (clip pour éviter valeurs négatives si nécessaire)
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

# Metriques numériques sécurisées
def safe_mape(y_true, y_pred):
    # Eviter division par zéro : ne calculer que sur y_true != 0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (y_true != 0)
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def safe_float(val):
    """Convertit val en float réel si possible, sinon retourne None."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().replace('%', '')
        try:
            v = float(s)
        except Exception:
            return None
    elif isinstance(val, complex):
        if val.imag != 0:
            return None
        v = float(val.real)
    else:
        try:
            v = float(val)
        except Exception:
            return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v

mae = safe_float(mean_absolute_error(y_test, y_pred))
rmse = safe_float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = safe_float(r2_score(y_test, y_pred))
mape = safe_mape(y_test, y_pred)
# Si MAPE None (ex: y_test all zeros), calculer SMAPE comme fallback
if mape is None:
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-8))
    mape = float(smape)
else:
    mape = float(mape)

metrics_raw = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'mape': mape,
    'n_test': int(len(y_test)),
    # 'model' removed from metrics (non numérique)
}

# Nettoyage des métriques avant envoi à MLflow
metrics_clean = {}
ignored = {}
for k, v in metrics_raw.items():
    v_clean = safe_float(v) if k != 'n_test' else (int(v) if v is not None else None)
    if v_clean is None:
        ignored[k] = v
    else:
        metrics_clean[k] = v_clean

log.info(f"MAE={mae:.2f}j | RMSE={rmse:.2f}j | R2={r2:.3f} | MAPE={mape:.1f}%")

# MLflow + graphiques
with mlflow.start_run():
    # Log model name as param (texte autorisé)
    mlflow.log_param('model', type(model).__name__)
    # Log numeric metrics only
    if ignored:
        log.warning(f"Some metrics were ignored and not logged to MLflow because they are not real numbers: {ignored}")
    if metrics_clean:
        mlflow.log_metrics(metrics_clean)
        log.info(f"Logged metrics to MLflow: {metrics_clean}")
    else:
        log.error("No valid numeric metrics to log to MLflow.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.2, s=10, color='steelblue')
    lim = max(float(np.nanmax(y_test)), float(np.nanmax(y_pred)))
    axes[0].plot([0, lim], [0, lim], 'r--', lw=2)
    axes[0].set(xlabel='Reel (jours)', ylabel='Predit (jours)',
               title=f'Predit vs Reel — MAE={mae:.2f}j')
    residuals = y_test.values - y_pred
    axes[1].hist(residuals, bins=50, color='steelblue', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set(xlabel='Residus (jours)', title=f'Residus — RMSE={rmse:.2f}j')
    plt.tight_layout()

    os.makedirs(args.eval_output, exist_ok=True)
    fig_path = f'{args.eval_output}/diagnostics.png'
    fig.savefig(fig_path, dpi=150)
    try:
        mlflow.log_artifact(fig_path)
    except Exception as e:
        log.warning(f"mlflow.log_artifact failed for diagnostics.png: {e}")
    plt.close()

    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp['feature'], imp['importance'])
        ax.set_title(f'Feature Importances — {type(model).__name__}')
        plt.tight_layout()
        fi_path = f'{args.eval_output}/feature_importance.png'
        fig2.savefig(fi_path, dpi=150)
        try:
            mlflow.log_artifact(fi_path)
        except Exception as e:
            log.warning(f"mlflow.log_artifact failed for feature_importance.png: {e}")
        plt.close()

    # Sauvegarde des métriques nettoyées sur disque (inclut aussi les valeurs brutes pour debug)
    with open(f'{args.eval_output}/metrics.json', 'w') as fh:
        json.dump({'metrics_logged': metrics_clean, 'metrics_raw': metrics_raw, 'ignored': ignored}, fh, indent=2)
    try:
        mlflow.log_artifact(f'{args.eval_output}/metrics.json')
    except Exception as e:
        log.warning(f"mlflow.log_artifact failed for metrics.json: {e}")

log.info('Evaluation terminee.')
