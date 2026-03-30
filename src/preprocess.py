# src/preprocess.py — Définition du composant

from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output 
from azure.ai.ml.constants import AssetTypes
import os

 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
 
preprocess_olist = CommandComponent(
    name='preprocess_olist',
    version='1',
    display_name='1 - Preprocess Olist',
    description='Jointure 3 CSV + nettoyage + calcul delivery_days',
    environment='azureml:olist-train-env:3',
    code=SRC_DIR,
    command=(
        'python preprocess_script.py '
        '--raw_orders ${{inputs.raw_orders}} '
        '--raw_items ${{inputs.raw_items}} '
        '--raw_products ${{inputs.raw_products}} '
        '--clean_output ${{outputs.clean_output}}'
    ),
    inputs={
        'raw_orders':   Input(type=AssetTypes.URI_FILE),
        'raw_items':    Input(type=AssetTypes.URI_FILE),
        'raw_products': Input(type=AssetTypes.URI_FILE),
    },
    outputs={
        'clean_output': Output(type=AssetTypes.URI_FOLDER),
    },
)