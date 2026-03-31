#register_model.py
#IMPORTANT : avant de lancer le script, faire ctrl+shift+P pour sélectionner l'env olist
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import json, os
 
ml_client = MLClient.from_config(DefaultAzureCredential())
 
PIPELINE_JOB = 'goofy_roti_jydxv7qq6r'   # Allez dans Studio -> Jobs puis copier le nom de votre job winner ici !
 
# Télécharger les métriques
os.makedirs('/tmp/metrics', exist_ok=True)
ml_client.jobs.download(name=PIPELINE_JOB,
    download_path='/tmp/metrics/', output_name='metrics')
with open('/tmp/metrics/named-outputs/metrics/metrics.json') as f:
    m = json.load(f)
 
# Enregistrer le modèle v2 (manuel)
model = ml_client.models.create_or_update(Model(
    path=f'azureml://jobs/{PIPELINE_JOB}/outputs/model/',
    name='olist-predictor',
    version='2',
    description=f'gradient_boosting — MAE={m["metrics_logged"]["mae"]:.2f}j R2={m["metrics_logged"]["r2"]:.3f}',
    tags={'mae':str(round(m["metrics_logged"]['mae'],2)),'r2':str(round(m["metrics_logged"]['r2'],3))},
    type=AssetTypes.CUSTOM_MODEL,  
))
print(f'Enregistre : {model.name} v{model.version}')
print('Studio -> Models -> olist-predictor')