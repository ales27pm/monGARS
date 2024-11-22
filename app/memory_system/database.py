from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

Base = declarative_base()
DATABASE_URL = "sqlite:///./memory.db"  # Example SQLite DB for local use
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class MemoryModel(Base):
    __tablename__ = "memories"

    id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    importance = Column(Float, default=0.5)
    feedback = relationship("FeedbackModel", back_populates="memory")

class FeedbackModel(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    memory_id = Column(String, ForeignKey("memories.id"), nullable=False)
    rating = Column(Integer, nullable=True)  # Rating between 1 and 5
    comments = Column(Text, nullable=True)
    memory = relationship("MemoryModel", back_populates="feedback")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()