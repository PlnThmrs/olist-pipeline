# src/score.py
# FEATURE_NAMES doit être IDENTIQUE à feature_engineering_script.py
import json, os, logging, joblib
import pandas as pd, numpy as np
 
model = None
scaler = None
 
FEATURE_NAMES = [
    'price','freight_value','product_weight_g',
    'product_length_cm','product_height_cm','product_width_cm',
    'product_volume_cm3','freight_ratio',
    'purchase_month','purchase_dayofweek','category_encoded',
]
 
def init():
    global model, scaler
    model_dir = os.getenv('AZUREML_MODEL_DIR','.')
    model  = joblib.load(os.path.join(model_dir,'model_output/model.pkl'))
    scaler_path = os.path.join(model_dir,'model_output/scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    logging.info(f'Endpoint pret — {type(model).__name__}')
 
def run(raw_data):
    try:
        payload = json.loads(raw_data)
        records = payload.get('data', payload)
        df = pd.DataFrame(records)
        for feat in FEATURE_NAMES:
            if feat not in df.columns: df[feat] = 0
        df = df[FEATURE_NAMES].fillna(0)
        if scaler is not None:
            df = pd.DataFrame(scaler.transform(df), columns=FEATURE_NAMES)
        preds = np.maximum(model.predict(df), 0)
        return json.dumps({
            'predictions': [round(float(p),1) for p in preds],
            'unit': 'jours', 'count': int(len(preds))})
    except Exception as e:
        return json.dumps({'error': str(e)})