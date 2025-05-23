import json
import joblib
import mlflow.client
import mlflow.sklearn
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import os

from modules.crud import get_feedback_data

load_dotenv(override=True)
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_REGISTRY_URI = MLFLOW_TRACKING_URI
MLFLOW_S3_ENDPOINT_URL = os.environ['MLFLOW_S3_ENDPOINT_URL']
STAGE = os.environ['STAGE']
EXPERIMENT_NAME = os.environ['EXPERIMENT_NAME']
REGISTERED_MODEL_NAME = os.environ['REGISTERED_MODEL_NAME']

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

mlflow_client = mlflow.MlflowClient()

def train(retrain: bool = False):
    # load data
    if retrain:
        train = pd.DataFrame(get_feedback_data(), columns=['id', 'target', 'timestamp', 'input_json', 'model_version'])
        train.drop(columns=['id', 'timestamp', 'model_version'], inplace=True)

        # convert input_json string to dictinary
        train['input_json'] = train['input_json'].apply(json.loads)

        train = pd.concat([train['input_json'].apply(pd.Series), train['target']], axis=1)
    else:
        train = pd.read_csv('data/train.csv')
    y = train['target']
    X = train.drop(columns='target')

    # start mlflow experiment
    with mlflow.start_run():
        alpha = 1.0
        model = Ridge(alpha=alpha)
        model.fit(X,y)

        # log parameters
        mlflow.log_param('alpha', alpha)

        # log metrics
        preds = model.predict(X)
        mse = mean_squared_error(y_true=y, y_pred=preds)
        mlflow.log_metric('mse', mse)
        print(f'Training Complete. MSE: {mse:.4f}')

        # save model
        joblib.dump(model, 'model.joblib')


        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X.head(1),
            signature=mlflow.models.infer_signature(X, preds)
        )
        latest_version = mlflow_client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0].version
        print(f'version: {latest_version}')

        return model, latest_version

if __name__ == '__main__':
    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument('--retrain', action="store_true", help="Uses feedback labeled data for retraining")
    args = parser.parse_args()

    model, latest_version  = train(retrain=args.retrain)

    # load test dataset
    test = pd.read_csv('data/test.csv')
    y = test['target']
    X = test.drop(columns=['target'])

    # log metrics
    new_preds = model.predict(X)
    new_mse = mean_squared_error(y, new_preds)
    mlflow.log_metric('mse', new_mse)
    print(f'Testing  MSE: {new_mse:.4f}')

    try:
        prod_model = mlflow.pyfunc.load_model(f'models:/{REGISTERED_MODEL_NAME}/{STAGE}')
        prod_preds = prod_model.predict(X)
        prod_mse = mean_squared_error(y, prod_preds)
    except Exception:
        prod_mse = float('inf')

    # promote new model if better
    if new_mse < prod_mse:
        mlflow_client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=latest_version,
                stage = STAGE,
                archive_existing_versions=True
        )
        print(f"Promoted version {latest_version} to {STAGE} (MSE improved from {prod_mse:.4f} to {new_mse:.4f})")
    else:
        print(f"Kept existing model. New MSE: {new_mse:.4f}, Production MSE: {prod_mse:.4f}")
    mlflow.log_metric("val_mse", new_mse)
