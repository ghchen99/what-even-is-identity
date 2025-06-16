from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from datetime import datetime

from app.schemas.biometric import EnrollmentResponse, VerificationResponse
from app.services.storage import StorageService
from app.services.biometric import BiometricService
from app.models.user import UserEnrollment, VerificationAttempt
from app.core.config import settings

router = APIRouter()

# Dependency injection
def get_storage_service() -> StorageService:
    return StorageService()

def get_biometric_service() -> BiometricService:
    return BiometricService()

@router.post("/enroll", response_model=EnrollmentResponse)
async def enroll_user(
    user_id: str = Form(...),
    face_image: UploadFile = File(...),
    voice_audio: UploadFile = File(...),
    storage: StorageService = Depends(get_storage_service),
    biometric: BiometricService = Depends(get_biometric_service)
):
    """
    Enroll a new user with face and voice biometrics
    """
    try:
        # Validate file sizes
        if face_image.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Face image file too large")
        if voice_audio.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Voice audio file too large")
        
        # Read file data
        face_data = await face_image.read()
        voice_data = await voice_audio.read()
        
        # Validate file formats
        if not biometric.validate_image_format(face_data):
            raise HTTPException(status_code=400, detail="Invalid face image format")
        if not biometric.validate_audio_format(voice_data):
            raise HTTPException(status_code=400, detail="Invalid voice audio format")
        
        # Extract embeddings
        face_embedding = await biometric.extract_face_embedding(face_data, f"enroll_{user_id}")
        voice_embedding = await biometric.extract_voice_embedding(voice_data)
        
        # Create enrollment record
        enrollment = UserEnrollment.create(
            user_id=user_id,
            face_embedding=face_embedding.tolist(),
            voice_embedding=voice_embedding.tolist()
        )
        
        # Save enrollment
        enrollment_id = await storage.save_user_enrollment(user_id, enrollment)
        
        return EnrollmentResponse(
            status="success",
            enrollment_id=enrollment_id,
            message=f"User {user_id} enrolled successfully",
            user_id=user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@router.post("/verify", response_model=VerificationResponse)
async def verify_user(
    user_id: str = Form(...),
    face_image: UploadFile = File(...),
    voice_audio: UploadFile = File(...),
    threshold: float = Form(default=settings.DEFAULT_SIMILARITY_THRESHOLD),
    storage: StorageService = Depends(get_storage_service),
    biometric: BiometricService = Depends(get_biometric_service)
):
    """
    Verify user identity using face and voice biometrics
    """
    try:
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
        
        # Get stored user enrollment
        enrollment = await storage.get_user_enrollment(user_id)
        if not enrollment:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Validate file sizes
        if face_image.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Face image file too large")
        if voice_audio.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Voice audio file too large")
        
        # Read file data
        face_data = await face_image.read()
        voice_data = await voice_audio.read()
        
        # Validate file formats
        if not biometric.validate_image_format(face_data):
            raise HTTPException(status_code=400, detail="Invalid face image format")
        if not biometric.validate_audio_format(voice_data):
            raise HTTPException(status_code=400, detail="Invalid voice audio format")
        
        # Extract embeddings from verification data
        face_embedding = await biometric.extract_face_embedding(face_data, f"verify_{user_id}")
        voice_embedding = await biometric.extract_voice_embedding(voice_data)
        
        # Get stored embeddings
        import numpy as np
        stored_face = np.array(enrollment.face_embedding)
        stored_voice = np.array(enrollment.voice_embedding)
        
        # Calculate similarities
        face_similarity = biometric.calculate_cosine_similarity(face_embedding, stored_face)
        voice_similarity = biometric.calculate_cosine_similarity(voice_embedding, stored_voice)
        
        # Calculate combined score
        combined_score = biometric.calculate_combined_score(face_similarity, voice_similarity)
        
        # Get separate thresholds from config
        face_threshold = settings.FACE_VERIFICATION_THRESHOLD
        voice_threshold = settings.VOICE_VERIFICATION_THRESHOLD
        
        # Create verification attempt record with dual verification
        verification = VerificationAttempt.create(
            user_id=user_id,
            face_similarity=face_similarity,
            voice_similarity=voice_similarity,
            combined_score=combined_score,
            threshold=threshold,
            face_threshold=face_threshold,
            voice_threshold=voice_threshold
        )
        
        # Save verification attempt
        verification_id = await storage.save_verification_attempt(verification)
        
        return VerificationResponse(
            status="success" if verification.verified else "failed",
            verified=verification.verified,
            face_similarity=face_similarity,
            voice_similarity=voice_similarity,
            combined_score=combined_score,
            threshold=threshold,
            verification_id=verification_id,
            face_verified=verification.face_verified,
            voice_verified=verification.voice_verified,
            face_threshold=face_threshold,
            voice_threshold=voice_threshold
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")