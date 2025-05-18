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

load_dotenv()
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_REGISTRY_URI = MLFLOW_TRACKING_URI
STAGE = os.environ['STAGE']

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
mlflow.set_experiment("diabetes")

# load data
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
    # log model
    registered_model_name='diabetes-ridge-model'

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        registered_model_name=registered_model_name
    )

    # get latest version
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version

    # promote to staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=latest_version,
        stage = STAGE,
        archive_existing_versions=True
    )
    print(f"Model version {latest_version} promoted to {STAGE}")
