# src/feature_engineering.py — Définition du composant
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
 
feature_engineering_olist = CommandComponent(
    name='feature_engineering_olist',
    version='1',
    display_name='2 - Feature Engineering Olist',
    description='11 features + split train/test + StandardScaler',
    environment='azureml:olist-train-env:3',
    code=SRC_DIR,
    command=(
        'python feature_engineering_script.py '
        '--clean_data ${{inputs.clean_data}} '
        '--train_output ${{outputs.train_output}} '
        '--test_output ${{outputs.test_output}} '
        '--test_size ${{inputs.test_size}} '
        '--random_state ${{inputs.random_state}}'
    ),
    inputs={
        'clean_data':   Input(type=AssetTypes.URI_FOLDER),
        'test_size':    Input(type='number',  default=0.2),
        'random_state': Input(type='integer', default=42),
    },
    outputs={
        'train_output': Output(type=AssetTypes.URI_FOLDER),
        'test_output':  Output(type=AssetTypes.URI_FOLDER),
    },
)
