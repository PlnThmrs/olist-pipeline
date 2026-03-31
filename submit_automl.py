# submit_automl.py
from azure.ai.ml import MLClient, Input
from azure.ai.ml.automl import regression, TabularFeaturizationSettings, TabularLimitSettings
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os
 
ml_client = MLClient.from_config(DefaultAzureCredential())
 
PIPELINE_JOB = 'goofy_roti_jydxv7qq6r'   # Allez dans Studio -> Jobs puis copier le nom de votre job pipeline ici !
 
# Trouver le step preprocessing dans les jobs enfants du pipeline
child_jobs = list(ml_client.jobs.list(parent_job_name=PIPELINE_JOB))
preprocess_job = next(
    (j for j in child_jobs if 'pre' in j.display_name.lower()), None
)
if preprocess_job is None:
    raise RuntimeError('Step preprocessing non trouvé dans les jobs enfants')
print(f'Step preprocessing : {preprocess_job.name}')
 
# Télécharger le clean_output (clean.csv avec delivery_days + toutes les features)
DOWNLOAD_PATH = '/tmp/automl_data'
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
ml_client.jobs.download(
    name=preprocess_job.name,
    download_path=DOWNLOAD_PATH,
    output_name='clean_output',
)
data_folder = f'{DOWNLOAD_PATH}/named-outputs/clean_output'
 
# Créer le fichier MLTable requis par AutoML
with open(f'{data_folder}/MLTable', 'w') as f:
    f.write(
        "paths:\n"
        "  - file: ./clean.csv\n"
        "transformations:\n"
        "  - read_delimited:\n"
        "      delimiter: ','\n"
        "      encoding: 'utf8'\n"
        "      header: all_files_same_headers\n"
    )
 
# Enregistrer comme data asset MLTable dans Azure ML (réutilise si déjà existant)
try:
    data_asset = ml_client.data.get(name='olist-clean-mltable', version='1')
    print('Data asset déjà existant, réutilisation.')
except Exception:
    data_asset = ml_client.data.create_or_update(Data(
        path=data_folder,
        name='olist-clean-mltable',
        version='1',
        type=AssetTypes.MLTABLE,
        description='Données nettoyées Olist — entrée AutoML',
    ))
print(f'Data asset enregistré : {data_asset.id}')
 
# Soumettre le job AutoML regression
automl_job = regression(
    target_column_name='delivery_days',
    training_data=Input(type='mltable', path=data_asset.id),
    primary_metric='normalized_root_mean_squared_error',
    compute='cluster-pauline',
    experiment_name='olist-delivery-days',
    featurization=TabularFeaturizationSettings(mode='auto'),
    limits=TabularLimitSettings(
        timeout_minutes=30,
        trial_timeout_minutes=6,
        max_trials=20,
        enable_early_termination=True,
    ),
)
submitted = ml_client.jobs.create_or_update(automl_job)
print(f'AutoML soumis : {submitted.studio_url}')