from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List

from app.schemas.biometric import UserEnrollmentsResponse, UserVerificationsResponse
from app.services.storage import StorageService

router = APIRouter()

# Dependency injection
def get_storage_service() -> StorageService:
    return StorageService()

@router.get("/{user_id}/enrollments", response_model=UserEnrollmentsResponse)
async def get_user_enrollments(
    user_id: str,
    storage: StorageService = Depends(get_storage_service)
):
    """
    Get all enrollments for a specific user
    """
    try:
        user = await storage.get_user(user_id)
        if not user:
            return UserEnrollmentsResponse(user_id=user_id, enrollments=[])
        
        # Convert enrollments to response format
        enrollments = []
        for enrollment in user.enrollments:
            enrollments.append({
                "user_id": enrollment.user_id,
                "enrollment_id": enrollment.enrollment_id,
                "enrolled_at": enrollment.enrolled_at,
                "face_embedding_size": len(enrollment.face_embedding),
                "voice_embedding_size": len(enrollment.voice_embedding)
            })
        
        return UserEnrollmentsResponse(
            user_id=user_id,
            enrollments=enrollments
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enrollments: {str(e)}")

@router.get("/{user_id}/verifications", response_model=UserVerificationsResponse)
async def get_user_verifications(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    storage: StorageService = Depends(get_storage_service)
):
    """
    Get recent verification attempts for a specific user
    """
    try:
        verifications = await storage.get_user_verifications(user_id, limit)
        
        # Convert to response format
        verification_list = []
        for verification in verifications:
            verification_list.append({
                "verification_id": verification.verification_id,
                "user_id": verification.user_id,
                "timestamp": verification.timestamp,
                "face_similarity": verification.face_similarity,
                "voice_similarity": verification.voice_similarity,
                "verified": verification.verified,
                "face_verified": verification.face_verified,
                "voice_verified": verification.voice_verified
            })
        
        return UserVerificationsResponse(
            user_id=user_id,
            verifications=verification_list,
            total_count=len(verification_list)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get verifications: {str(e)}")

@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    storage: StorageService = Depends(get_storage_service)
):
    """
    Delete all data for a specific user
    """
    try:
        deleted = await storage.delete_user_data(user_id)
        if deleted:
            return {
                "status": "success",
                "message": f"User {user_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")