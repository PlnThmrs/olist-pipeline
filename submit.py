# submit.py 
# # Soumettre le pipeline Olist sur Azure ML 
# # Usage : python3 submit.py (depuis le terminal VS Code) 

from azure.ai.ml import MLClient, Input 
from azure.ai.ml.constants import AssetTypes 
from azure.identity import DefaultAzureCredential 
from pipeline import olist_pipeline 
import datetime 

def main(): 
    ml_client = MLClient.from_config(DefaultAzureCredential()) 
    print(f'Workspace : {ml_client.workspace_name}') 
    
    # ── Récupérer les 3 Data Assets ────────────────────────────────────── 
    
    def get(name, version='1'): 
        a = ml_client.data.get(name, version=version)
        return Input(type=AssetTypes.URI_FILE, path=a.path) 
    
    # ── Vos choix d'hyperparamètres ─────────────────────────────────────── 
    MODEL_TYPE = 'random_forest' # random_forest | gradient_boosting | lightgbm | ridge 
    N_ESTIMATORS = 100 
    MAX_DEPTH = 10 
    LEARN_RATE = 0.1 
    
    # ── Instanciation ───────────────────────────────────────────────────── 
    job = olist_pipeline(
        raw_orders=get('olist-orders'), 
        raw_items=get('olist-order-items'), 
        raw_products=get('olist-products'), 
        model_type=MODEL_TYPE, 
        n_estimators=N_ESTIMATORS, 
        max_depth=MAX_DEPTH, 
        learning_rate=LEARN_RATE
        ) 
    ts = datetime.datetime.now().strftime('%m%d-%H%M') 
    job.experiment_name = 'olist-delivery-days' 
    job.display_name = f'{MODEL_TYPE}-n{N_ESTIMATORS}-d{MAX_DEPTH}-{ts}' 
    
    # ── Soumission ──────────────────────────────────────────────────────── 
    submitted = ml_client.jobs.create_or_update(job) 
    print(f'Pipeline soumis : {submitted.name}') 
    print(f'Studio URL : {submitted.studio_url}') 
    print('Ouvrir l\'URL pour suivre le graphe en temps réel.') 
    print('Durée estimée : 10-20 min (première fois plus long — Docker build)') 
    
if __name__ == '__main__': 
    main()