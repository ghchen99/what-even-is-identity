from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class BiometricEmbedding:
    face_embedding: List[float]
    voice_embedding: List[float]
    created_at: datetime

@dataclass
class UserEnrollment:
    user_id: str
    enrollment_id: str
    face_embedding: List[float]
    voice_embedding: List[float]
    enrolled_at: datetime
    
    @classmethod
    def create(cls, user_id: str, face_embedding: List[float], voice_embedding: List[float]) -> 'UserEnrollment':
        return cls(
            user_id=user_id,
            enrollment_id=str(uuid.uuid4()),
            face_embedding=face_embedding,
            voice_embedding=voice_embedding,
            enrolled_at=datetime.now()
        )
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'enrolled_at': self.enrolled_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserEnrollment':
        data['enrolled_at'] = datetime.fromisoformat(data['enrolled_at'])
        return cls(**data)

@dataclass
class VerificationAttempt:
    verification_id: str
    user_id: str
    face_similarity: float
    voice_similarity: float
    combined_score: float
    threshold: float
    verified: bool
    timestamp: datetime
    # Dual verification fields
    face_verified: bool = False
    voice_verified: bool = False
    face_threshold: float = 0.80
    voice_threshold: float = 0.95
    
    @classmethod
    def create(
        cls, 
        user_id: str, 
        face_similarity: float, 
        voice_similarity: float, 
        combined_score: float, 
        threshold: float,
        face_threshold: float = 0.80,
        voice_threshold: float = 0.95
    ) -> 'VerificationAttempt':
        face_verified = face_similarity >= face_threshold
        voice_verified = voice_similarity >= voice_threshold
        
        return cls(
            verification_id=str(uuid.uuid4()),
            user_id=user_id,
            face_similarity=face_similarity,
            voice_similarity=voice_similarity,
            combined_score=combined_score,
            threshold=threshold,
            verified=face_verified and voice_verified,  # Both must pass for dual verification
            timestamp=datetime.now(),
            face_verified=face_verified,
            voice_verified=voice_verified,
            face_threshold=face_threshold,
            voice_threshold=voice_threshold
        )
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VerificationAttempt':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class User:
    user_id: str
    created_at: datetime
    enrollments: List[UserEnrollment]
    
    @classmethod
    def create(cls, user_id: str) -> 'User':
        return cls(
            user_id=user_id,
            created_at=datetime.now(),
            enrollments=[]
        )
    
    def add_enrollment(self, enrollment: UserEnrollment) -> None:
        self.enrollments.append(enrollment)
    
    def get_latest_enrollment(self) -> Optional[UserEnrollment]:
        if not self.enrollments:
            return None
        return max(self.enrollments, key=lambda e: e.enrolled_at)
    
    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'enrollments': [e.to_dict() for e in self.enrollments]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['enrollments'] = [UserEnrollment.from_dict(e) for e in data['enrollments']]
        return cls(**data)