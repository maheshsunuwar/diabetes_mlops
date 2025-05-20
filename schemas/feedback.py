from pydantic import BaseModel
from uuid import UUID

class FeedbackCreate(BaseModel):
    id: str
    correct: bool
