import joblib
import mlflow.sklearn
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


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

    # save model
    joblib.dump(model, 'model.joblib')
    mlflow.sklearn.log_model(model, 'model')

    print(f'Training Complete. MSE: {mse:.4f}')
