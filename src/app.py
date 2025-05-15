from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

class DiabetesInput(BaseModel):
    age: float
    sex: float
    bmi : float
    bp : float
    s1 : float
    s2 : float
    s3 : float
    s4 : float
    s5 : float
    s6 : float

# load model from mlflow
model_name = 'diabetes-ridge-model'
# # model_stage = 'None', #'Staging' # or 'Production'
# model_version = '2'
# model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_version}')

run_id = '63b603e6533d4c00979dcacec0684a5a'
model = mlflow.pyfunc.load_model(f'../mlruns/0/{run_id}/artifacts/model')

@app.get('/health')
def health():
    return {
        'status':'ok'
    }

@app.post('/predict')
def predict(data: DiabetesInput):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    return {
        'prediction': prediction.tolist()[0]
    }
