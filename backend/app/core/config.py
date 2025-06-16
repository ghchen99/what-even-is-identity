from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

# Get the backend directory (where this config file is located)
BACKEND_DIR = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # Project info
    PROJECT_NAME: str = "Biometric Unlock API"
    PROJECT_VERSION: str = "1.0.0"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["*"]  # In production, specify actual origins
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Storage settings
    DATA_DIR: Path = BACKEND_DIR / "data"
    MODELS_DIR: Path = BACKEND_DIR / "models"
    
    # Model settings
    FACE_MODEL_PATH: str = "facenet.onnx"
    VOICE_MODEL_PATH: str = "xvector.onnx"
    
    # Biometric settings  
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.8
    FACE_WEIGHT: float = 0.6
    VOICE_WEIGHT: float = 0.4
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    MAX_VERIFICATIONS_HISTORY: int = 100
    
    # Debug settings
    BIOMETRIC_DEBUG: bool = False
    
    # Security settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_AUDIO_TYPES: List[str] = ["audio/wav", "audio/mp3", "audio/m4a"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()