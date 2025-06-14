from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class EnrollmentRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=50, description="Unique user identifier")

class EnrollmentResponse(BaseModel):
    status: str
    enrollment_id: str
    message: str
    user_id: str

class VerificationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=50, description="User identifier")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold")

class VerificationResponse(BaseModel):
    status: str
    verified: bool
    face_similarity: float
    voice_similarity: float
    combined_score: float
    threshold: float
    verification_id: str

class BiometricScores(BaseModel):
    face_similarity: float = Field(..., ge=0.0, le=1.0)
    voice_similarity: float = Field(..., ge=0.0, le=1.0)
    combined_score: float = Field(..., ge=0.0, le=1.0)

class UserEnrollment(BaseModel):
    user_id: str
    enrollment_id: str
    enrolled_at: datetime
    face_embedding_size: int
    voice_embedding_size: int

class VerificationAttempt(BaseModel):
    verification_id: str
    user_id: str
    timestamp: datetime
    face_similarity: float
    voice_similarity: float
    combined_score: float
    threshold: float
    verified: bool

class UserEnrollmentsResponse(BaseModel):
    user_id: str
    enrollments: List[UserEnrollment]

class UserVerificationsResponse(BaseModel):
    user_id: str
    verifications: List[VerificationAttempt]
    total_count: int

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    models: dict

class ModelInfo(BaseModel):
    face_model: dict
    voice_model: dict

class CacheCleanupResponse(BaseModel):
    status: str
    cleaned_entries: int
    message: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None