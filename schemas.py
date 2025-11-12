from pydantic import BaseModel, Field
from typing import Optional

class Prediction(BaseModel):
    """Predictions collection schema
    Collection name: "prediction"
    """
    filename: str = Field(..., description="Original upload filename")
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence for predicted label")
    prob_cancer: float = Field(..., ge=0, le=1, description="Probability of cancer class")
    timestamp: str = Field(..., description="ISO timestamp when prediction was made")

# You can extend with user/session, modality, additional metadata later
