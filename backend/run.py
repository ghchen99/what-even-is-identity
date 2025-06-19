#!/usr/bin/env python3
"""
Biometric Unlock API Server
Entry point for running the FastAPI server
"""

import uvicorn
import os
import sys
from pathlib import Path

# Ensure the backend directory is in the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings

def main():
    """Run the FastAPI server"""
    
    print("🚀 Starting Biometric Unlock API Server...")
    print(f"📁 Data directory: {settings.DATA_DIR.absolute()}")
    print(f"🤖 Models directory: {settings.MODELS_DIR.absolute()}")
    print("💡 Place your ONNX models in the models/ directory:")
    print(f"   - {settings.FACE_MODEL_PATH} (face recognition)")
    print(f"   - {settings.VOICE_MODEL_PATH} (voice recognition)")
    print()
    print("📚 API Documentation:")
    print(f"   - http://{settings.HOST}:{settings.PORT}/docs (Swagger UI)")
    print(f"   - http://{settings.HOST}:{settings.PORT}/redoc (ReDoc)")
    print()
    print(f"🔍 Health check: http://{settings.HOST}:{settings.PORT}/api/v1/health")
    print()
    
    # Check if we're in production mode
    if os.getenv("PRODUCTION", "false").lower() == "true":
        settings.RELOAD = False
        print("🏭 Running in PRODUCTION mode")
    else:
        print("🛠️  Running in DEVELOPMENT mode (auto-reload enabled)")
    
    print(f"🌐 Server starting on http://{settings.HOST}:{settings.PORT}")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.RELOAD,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()