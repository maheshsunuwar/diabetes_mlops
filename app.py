from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI()
# mlflow.set_tracking_uri("http://host.docker.internal:4999")
# mlflow.set_tracking_uri("http://localhost:4999")
mlflow.set_tracking_uri("http://mlflow:5000")

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
model_stage = 'Staging' # or 'Production'
# model_version = '1'
model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_stage}')

# run_id = '0760de80da5e485eb1ecabb96dd4adbc'
# model = mlflow.pyfunc.load_model(f'/app/mlruns/1/{run_id}/artifacts/model')

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

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000)
