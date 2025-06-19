from fastapi import APIRouter, Depends
from datetime import datetime

from app.schemas.biometric import HealthCheck, ModelInfo, CacheCleanupResponse
from app.services.storage import StorageService
from app.services.biometric import BiometricService

router = APIRouter()

# Dependency injection
def get_storage_service() -> StorageService:
    return StorageService()

def get_biometric_service() -> BiometricService:
    return BiometricService()

@router.get("/", response_model=HealthCheck)
async def health_check(
    biometric: BiometricService = Depends(get_biometric_service)
):
    """
    Health check endpoint with model status information
    """
    model_info = biometric.get_model_info()
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        models=model_info
    )

@router.get("/models", response_model=ModelInfo)
async def get_model_info(
    biometric: BiometricService = Depends(get_biometric_service)
):
    """
    Get detailed information about loaded ONNX models
    """
    return ModelInfo(**biometric.get_model_info())

@router.post("/cache/cleanup", response_model=CacheCleanupResponse)
async def cleanup_cache(
    storage: StorageService = Depends(get_storage_service)
):
    """
    Clean up expired cache entries
    """
    try:
        cleaned_count = await storage.cleanup_expired_cache()
        return CacheCleanupResponse(
            status="success",
            cleaned_entries=cleaned_count,
            message=f"Cleaned {cleaned_count} expired cache entries"
        )
    except Exception as e:
        return CacheCleanupResponse(
            status="error",
            cleaned_entries=0,
            message=f"Cache cleanup failed: {str(e)}"
        )