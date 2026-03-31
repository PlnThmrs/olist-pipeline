# deploy_endpoint.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, ManagedOnlineDeployment,
    Environment, CodeConfiguration)
from azure.identity import DefaultAzureCredential
import os
 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
 
ml_client = MLClient.from_config(DefaultAzureCredential())
 
# Créer l'endpoint
ep = ManagedOnlineEndpoint(name='olist-endpoint', auth_mode='key')
ep = ml_client.online_endpoints.begin_create_or_update(ep).result()
print(f'Endpoint créé : {ep.name}')
 
# Déployer le modèle du Registry
deploy = ManagedOnlineDeployment(
    name='blue',
    endpoint_name='olist-endpoint',
    model='azureml:olist-predictor:2',
    code_configuration=CodeConfiguration(code=f'{PROJECT_ROOT}/src', scoring_script='score.py'),
    environment=Environment(
        name='olist-deploy-env',
        conda_file=f'{PROJECT_ROOT}/conda.yml',
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest',
    ),
    instance_type='Standard_DS2_v2',
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deploy).result()
print('Déploiement blue terminé')
 
# Router 100% du traffic vers blue
ep.traffic = {'blue': 100}
ml_client.online_endpoints.begin_create_or_update(ep).result()
 
# Afficher les infos de connexion
ep   = ml_client.online_endpoints.get('olist-endpoint')
keys = ml_client.online_endpoints.get_keys('olist-endpoint')
print(f'URL : {ep.scoring_uri}')
print(f'Key : {keys.primary_key[:30]}...')