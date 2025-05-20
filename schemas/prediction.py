from typing import Dict
from pydantic import BaseModel


class PredictionCreate(BaseModel):
    input_json: Dict
