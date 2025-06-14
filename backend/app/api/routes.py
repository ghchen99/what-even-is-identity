from fastapi import APIRouter
from app.api.endpoints import biometric, health, users

api_router = APIRouter()

api_router.include_router(biometric.router, prefix="/biometric", tags=["biometric"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(health.router, prefix="/health", tags=["health"])