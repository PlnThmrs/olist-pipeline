# test_endpoint.py
# Teste l'endpoint olist-endpoint après déploiement
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json, requests
 
ml_client = MLClient.from_config(DefaultAzureCredential())
 
# Récupérer l'URL et la clé automatiquement
ep   = ml_client.online_endpoints.get('olist-endpoint')
keys = ml_client.online_endpoints.get_keys('olist-endpoint')
 
URL = ep.scoring_uri
KEY = keys.primary_key
 
print(f'URL : {URL}')
print(f'Key : {KEY[:30]}...\n')
 
# Payload de test — 3 commandes fictives
payload = {
    "data": [
        {
            "price": 120.0,
            "freight_value": 15.0,
            "product_weight_g": 800,
            "product_length_cm": 30,
            "product_height_cm": 20,
            "product_width_cm": 15,
            "product_volume_cm3": 9000,
            "freight_ratio": 0.125,
            "purchase_month": 6,
            "purchase_dayofweek": 2,
            "category_encoded": 5,
        },
        {
            "price": 45.0,
            "freight_value": 8.5,
            "product_weight_g": 300,
            "product_length_cm": 20,
            "product_height_cm": 10,
            "product_width_cm": 10,
            "product_volume_cm3": 2000,
            "freight_ratio": 0.189,
            "purchase_month": 11,
            "purchase_dayofweek": 4,
            "category_encoded": 12,
        },
        {
            "price": 350.0,
            "freight_value": 40.0,
            "product_weight_g": 5000,
            "product_length_cm": 60,
            "product_height_cm": 40,
            "product_width_cm": 30,
            "product_volume_cm3": 72000,
            "freight_ratio": 0.114,
            "purchase_month": 1,
            "purchase_dayofweek": 0,
            "category_encoded": 3,
        },
    ]
}
 
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}",
}
 
response = requests.post(URL, headers=headers, json=payload)
 
print(f'Status  : {response.status_code}')
print(f'Réponse : {json.dumps(response.json(), indent=2, ensure_ascii=False)}')

if response.status_code == 200:
    result = json.loads(response.json())
    preds = result.get('predictions', [])
    print(f'\n--- Résultats ---')
    for i, days in enumerate(preds,1):
        print(f'  Commande {i} : {days} jours de livraison estimés')
else:
    print(f'Erreur : {response.text}')