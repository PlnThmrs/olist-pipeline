# src/feature_engineering_script.py
import argparse, os, logging, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
parser = argparse.ArgumentParser()
parser.add_argument('--clean_data',   type=str, required=True)
parser.add_argument('--train_output', type=str, required=True)
parser.add_argument('--test_output',  type=str, required=True)
parser.add_argument('--test_size',    type=float, default=0.2)
parser.add_argument('--random_state', type=int,   default=42)
args = parser.parse_args()
 
df = pd.read_csv(f'{args.clean_data}/clean.csv')
log.info(f'Donnees chargees : {df.shape}')
 
# Features temporelles
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['purchase_month']     = df['order_purchase_timestamp'].dt.month
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
 
# Features derivees
df['product_volume_cm3'] = (
    df['product_length_cm']
    * df['product_height_cm']
    * df['product_width_cm']
)
df['freight_ratio'] = df['freight_value'] / (df['price'] + 1e-6)
 
# Encodage categoriel
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(
    df['product_category_name'].fillna('unknown')
)
 
# Selection des features
FEATURE_NAMES = [
    'price', 'freight_value',
    'product_weight_g', 'product_length_cm',
    'product_height_cm', 'product_width_cm',
    'product_volume_cm3', 'freight_ratio',
    'purchase_month', 'purchase_dayofweek',
    'category_encoded',
]
 
X = df[FEATURE_NAMES].fillna(0)
y = df['delivery_days']
 
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)
log.info(f'Train : {X_train.shape} | Test : {X_test.shape}')
 
# Normalisation
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_NAMES)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_NAMES)
 
# Sauvegarde
os.makedirs(args.train_output, exist_ok=True)
X_train_s.to_csv(f'{args.train_output}/X_train.csv', index=False)
y_train.to_csv(  f'{args.train_output}/y_train.csv', index=False)
joblib.dump(scaler, f'{args.train_output}/scaler.pkl')
with open(f'{args.train_output}/feature_names.txt', 'w') as fh:
    fh.write('\n'.join(FEATURE_NAMES))
 
os.makedirs(args.test_output, exist_ok=True)
X_test_s.to_csv(f'{args.test_output}/X_test.csv', index=False)
y_test.to_csv(  f'{args.test_output}/y_test.csv', index=False)
 
log.info('Feature engineering termine.')
