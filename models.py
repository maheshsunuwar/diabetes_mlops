from sqlalchemy import Column, Float, Integer, DateTime, String, UUID
from datetime import datetime
from db import Base


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(UUID(as_uuid=True), primary_key=True, index=False)
    timestamp = Column(DateTime, default=datetime.now())
    input_json = Column(String)
    prediction = Column(Float)
    model_version = Column(String)
