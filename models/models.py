from sqlalchemy import Column, Float, ForeignKey, Integer, DateTime, String, UUID, Boolean
import uuid
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.now(), nullable=False)
    input_json = Column(String)
    prediction = Column(Float)
    model_version = Column(String)

class Feedback(Base):
    __tablename__ = 'feedbacks'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey('predictions.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    correct = Column(Boolean, nullable=False)

    predictions = relationship('Prediction', backref='feedbacks')
