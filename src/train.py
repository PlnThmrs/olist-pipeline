# src/train.py — Définition du composant
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
 
train_olist = CommandComponent(
    name='train_olist',
    version='1',
    display_name='3 - Train Olist Model',
    description='Entrainement multi-algo avec MLflow tracking',
    environment='azureml:olist-train-env:3',
    code=SRC_DIR,
    command=(
        'python train_script.py '
        '--train_data ${{inputs.train_data}} '
        '--model_type ${{inputs.model_type}} '
        '--n_estimators ${{inputs.n_estimators}} '
        '--max_depth ${{inputs.max_depth}} '
        '--learning_rate ${{inputs.learning_rate}} '
        '--model_output ${{outputs.model_output}}'
    ),
    inputs={
        'train_data':    Input(type=AssetTypes.URI_FOLDER),
        'model_type':    Input(type='string',  default='random_forest'),
        'n_estimators':  Input(type='integer', default=100),
        'max_depth':     Input(type='integer', default=10),
        'learning_rate': Input(type='number',  default=0.1),
    },
    outputs={
        'model_output': Output(type=AssetTypes.URI_FOLDER),
    },
)