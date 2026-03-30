# src/preprocess_script.py — Code de nettoyage
import argparse, os, logging
import pandas as pd
import numpy as np
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
parser = argparse.ArgumentParser()
parser.add_argument('--raw_orders',   type=str, required=True)
parser.add_argument('--raw_items',    type=str, required=True)
parser.add_argument('--raw_products', type=str, required=True)
parser.add_argument('--clean_output', type=str, required=True)
args = parser.parse_args()
 
# 1. Chargement
log.info('Chargement des CSV...')
orders   = pd.read_csv(args.raw_orders)
items    = pd.read_csv(args.raw_items)
products = pd.read_csv(args.raw_products)
log.info(f'Orders:{len(orders):,} | Items:{len(items):,} | Products:{len(products):,}')
 
# 2. Jointures
df = (orders
      .merge(items,    on='order_id',   how='inner')
      .merge(products, on='product_id', how='left'))
log.info(f'Apres jointures : {len(df):,} lignes')
 
# 3. Filtre : commandes livrees uniquement
df = df[df['order_status'] == 'delivered'].copy()
 
# 4. Variable cible : delivery_days
df['order_purchase_timestamp']      = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['delivery_days'] = (
    df['order_delivered_customer_date']
    - df['order_purchase_timestamp']
).dt.days
 
# 5. Filtres qualite
df = df[(df['delivery_days'] >= 0) & (df['delivery_days'] <= 120)]
df = df[(df['price'] > 0) & (df['freight_value'] >= 0)]
df = df.drop_duplicates(subset=['order_id', 'order_item_id'])
 
# 6. Colonnes utiles
keep_cols = [
    'order_id', 'order_item_id', 'order_purchase_timestamp',
    'price', 'freight_value',
    'product_weight_g', 'product_length_cm',
    'product_height_cm', 'product_width_cm',
    'product_category_name', 'delivery_days',
]
df = df[keep_cols].dropna(subset=['delivery_days', 'price', 'product_weight_g'])
 
log.info(f'Shape finale : {df.shape}')
log.info(f'delivery_days moyen : {df["delivery_days"].mean():.1f} jours')
 
# 7. Sauvegarde
os.makedirs(args.clean_output, exist_ok=True)
df.to_csv(f'{args.clean_output}/clean.csv', index=False)
log.info(f'Sauvegarde -> {args.clean_output}/clean.csv')
