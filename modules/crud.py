from fastapi import Depends
from requests import Session
from db import get_db
from models.models import Feedback, Prediction
db: Session = next(get_db())

def get_feedback_data():
    feedbacks = (
        db.query(
            Feedback.id,
            Feedback.correct,
            Feedback.timestamp,
            Prediction.input_json,
            Prediction.model_version
        )
        .join(Prediction, Prediction.id==Feedback.prediction_id)
        .order_by(Feedback.timestamp.desc())
        # .limit(100)
        .all()
    )

    return feedbacks
