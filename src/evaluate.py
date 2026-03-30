# src/evaluate.py — Définition du composant
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
 
evaluate_olist = CommandComponent(
    name='evaluate_olist',
    version='1',
    display_name='4 - Evaluate Olist Model',
    description='MAE, RMSE, R2, MAPE + graphiques + rapport JSON',
    environment='azureml:olist-train-env:3',
    code=SRC_DIR,
    command=(
        'python evaluate_script.py '
        '--model_input ${{inputs.model_input}} '
        '--test_data ${{inputs.test_data}} '
        '--eval_output ${{outputs.eval_output}}'
    ),
    inputs={
        'model_input': Input(type=AssetTypes.URI_FOLDER),
        'test_data':   Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={
        'eval_output': Output(type=AssetTypes.URI_FOLDER),
    },
)
