from typing import Dict
from pydantic import BaseModel
from uuid import UUID

class FeedbackCreate(BaseModel):
    id: str
    correct: bool
class FeedbackDetail(BaseModel):
    id: str
    correct: bool
    timestamp: str
    input_json: str
    model_version: str
