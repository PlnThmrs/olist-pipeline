# pipeline.py — Version compatible CommandComponent
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
 
# Les composants sont importes comme des objets
from src.preprocess          import preprocess_olist
from src.feature_engineering import feature_engineering_olist
from src.train               import train_olist
from src.evaluate            import evaluate_olist
 
 
@pipeline(
    name='olist_delivery_pipeline',
    description='Pipeline Olist : preprocess -> features -> train -> evaluate',
    compute='cluster-pauline',
)
def olist_pipeline(
    raw_orders,
    raw_items,
    raw_products,
    model_type='random_forest',
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
):
    # Etape 1 : Nettoyage
    step_pre = preprocess_olist(
        raw_orders=raw_orders,
        raw_items=raw_items,
        raw_products=raw_products,
    )
    # Etape 2 : Feature Engineering
    step_fe = feature_engineering_olist(
        clean_data=step_pre.outputs.clean_output,
    )
    # Etape 3 : Entrainement
    step_train = train_olist(
        train_data=step_fe.outputs.train_output,
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    # Etape 4 : Evaluation
    step_eval = evaluate_olist(
        model_input=step_train.outputs.model_output,
        test_data=step_fe.outputs.test_output,
    )
    return {
        'model':   step_train.outputs.model_output,
        'metrics': step_eval.outputs.eval_output,
    }
