from contextlib import asynccontextmanager
import json
from uuid import uuid4, UUID
from fastapi import Depends, FastAPI, HTTPException, Header
import mlflow
import pandas as pd
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

from sqlalchemy.orm import Session
from db import DbSession, get_db, engine
from models.models import Base, Feedback, Prediction
from schemas.feedback import FeedbackCreate

load_dotenv(override=True)
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
STAGE=os.environ.get('STAGE', 'Production')
APP_API_KEY = os.getenv("APP_API_KEY")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

#create tables if it doesnot exist
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifesapan(app: FastAPI):
    load_model()
    yield

app = FastAPI(lifespan=lifesapan)

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

def verify_api_key(x_api_key: str=Header(...)):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail='Invalid or mission api key.')

def load_model():
    """Loads the latest model.

    Returns:
        bool: True or False
    """
    model_name = 'diabetes-ridge-model'
    global model
    try:
        model = mlflow.pyfunc.load_model(f'models:/{model_name}/{STAGE}')
        print('Model loaded successfully.')
        return True
    except mlflow.MlflowException as e:
        print(f'Failed to load model: {e}')
        model = None
        return False

@app.get('/health')
def health():
    return {
        'status':'ok'
    }

@app.post('/predict')
def predict(data: DiabetesInput, auth:str=Depends(verify_api_key), db: Session = Depends(get_db)):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data).tolist()[0]

    db_prediction = Prediction(
        id = uuid4(),
        input_json=json.dumps(data.dict()),
        prediction=prediction,
        model_version=STAGE
    )

    db.add(db_prediction)
    db.commit()

    return {
        'prediction' : prediction,
        'id': db_prediction.id
    }

@app.post('/log_feedback')
def log_feedback(data: FeedbackCreate, auth:str=Depends(verify_api_key), db: Session= Depends(get_db)):
    db_feedback = Feedback(
        id = uuid4(),
        prediction_id = UUID(data.id),
        correct = data.correct
    )
    db.add(db_feedback)
    db.commit()

    return {
        'status':'ok'
    }


@app.post('/reload')
def reload_model(auth: str= Depends(verify_api_key)):
    success = load_model()
    if success:
        return {
            'status': 'Model reloaded.'
        }
    raise HTTPException(status_code=500, detail='Reload Failed')


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=9003)
