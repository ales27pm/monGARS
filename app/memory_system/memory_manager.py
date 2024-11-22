from sklearn.semi_supervised import LabelSpreading
from sqlalchemy.orm import Session
from .database import MemoryModel, FeedbackModel

class MemoryManager:
    def __init__(self, db: Session):
        self.db = db

    def calculate_feedback_score(self):
        feedbacks = self.db.query(FeedbackModel).all()
        if not feedbacks:
            return {"average_rating": None, "total_feedbacks": 0}

        total_rating = sum(fb.rating for fb in feedbacks if fb.rating)
        total_feedbacks = len([fb for fb in feedbacks if fb.rating])

        return {
            "average_rating": total_rating / total_feedbacks if total_feedbacks else 0,
            "total_feedbacks": total_feedbacks,
        }

    def calculate_cluster_quality(self, embeddings, labels):
        if len(set(labels)) < 2:
            return {"silhouette_score": None}

        from sklearn.metrics import silhouette_score
        score = silhouette_score(embeddings, labels)
        return {"silhouette_score": score}