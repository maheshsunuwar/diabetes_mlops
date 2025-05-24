from fastapi import Depends
from requests import Session
from db import get_db
from models.models import Feedback, Prediction
from dotenv import load_dotenv
load_dotenv(override=True)
db: Session = next(get_db())

def get_feedback_data():
    feedbacks = (
        db.query(
            Feedback.id,
            Feedback.correct,
            Feedback.timestamp,
            Prediction.input_json,
            Prediction.model_version,
            Prediction.prediction
        )
        .join(Prediction, Prediction.id==Feedback.prediction_id)
        .order_by(Feedback.timestamp.desc())
        # .limit(100)
        .all()
    )

    return feedbacks

def get_feedback_summary():
    total = db.query(Feedback).count()
    correct = db.query(Feedback).filter(Feedback.correct == True).count()
    incorrect = db.query(Feedback).filter(Feedback.correct == False).count()

    accuracy = round((correct/total)*100, 2) if total >9 else 0.0

    return (total, correct, incorrect, accuracy)
