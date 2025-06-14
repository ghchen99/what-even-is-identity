import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.models.user import User, UserEnrollment, VerificationAttempt

class StorageService:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.users_dir = self.data_dir / "users"
        self.verifications_dir = self.data_dir / "verifications"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.verifications_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_user_file_path(self, user_id: str) -> Path:
        return self.users_dir / f"{user_id}.json"
    
    def _get_verifications_file_path(self, user_id: str) -> Path:
        return self.verifications_dir / f"{user_id}_verifications.json"
    
    async def save_user_enrollment(self, user_id: str, enrollment: UserEnrollment) -> str:
        """Save user enrollment data"""
        user_file = self._get_user_file_path(user_id)
        
        if user_file.exists():
            # Load existing user
            with open(user_file, 'r') as f:
                user_data = json.load(f)
            user = User.from_dict(user_data)
        else:
            # Create new user
            user = User.create(user_id)
        
        user.add_enrollment(enrollment)
        
        # Save updated user
        with open(user_file, 'w') as f:
            json.dump(user.to_dict(), f, indent=2)
        
        return enrollment.enrollment_id
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user data"""
        user_file = self._get_user_file_path(user_id)
        
        if not user_file.exists():
            return None
        
        with open(user_file, 'r') as f:
            user_data = json.load(f)
        
        return User.from_dict(user_data)
    
    async def get_user_enrollment(self, user_id: str) -> Optional[UserEnrollment]:
        """Get latest user enrollment"""
        user = await self.get_user(user_id)
        if not user:
            return None
        
        return user.get_latest_enrollment()
    
    async def save_verification_attempt(self, attempt: VerificationAttempt) -> str:
        """Save verification attempt"""
        verifications_file = self._get_verifications_file_path(attempt.user_id)
        
        # Load existing verifications
        if verifications_file.exists():
            with open(verifications_file, 'r') as f:
                verifications_data = json.load(f)
        else:
            verifications_data = []
        
        # Add new verification
        verifications_data.append(attempt.to_dict())
        
        # Keep only recent verifications
        if len(verifications_data) > settings.MAX_VERIFICATIONS_HISTORY:
            verifications_data = verifications_data[-settings.MAX_VERIFICATIONS_HISTORY:]
        
        # Save updated verifications
        with open(verifications_file, 'w') as f:
            json.dump(verifications_data, f, indent=2)
        
        return attempt.verification_id
    
    async def get_user_verifications(self, user_id: str, limit: int = 10) -> List[VerificationAttempt]:
        """Get recent verification attempts"""
        verifications_file = self._get_verifications_file_path(user_id)
        
        if not verifications_file.exists():
            return []
        
        with open(verifications_file, 'r') as f:
            verifications_data = json.load(f)
        
        # Convert to objects and sort by timestamp
        verifications = [VerificationAttempt.from_dict(v) for v in verifications_data]
        verifications.sort(key=lambda v: v.timestamp, reverse=True)
        
        return verifications[:limit]
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data"""
        user_file = self._get_user_file_path(user_id)
        verifications_file = self._get_verifications_file_path(user_id)
        
        deleted = False
        
        if user_file.exists():
            user_file.unlink()
            deleted = True
        
        if verifications_file.exists():
            verifications_file.unlink()
            deleted = True
        
        return deleted
    
    # Cache operations (mock Redis)
    async def cache_set(self, key: str, value: any, ttl: int = None) -> None:
        """Set cache value"""
        if ttl is None:
            ttl = settings.CACHE_TTL
        
        cache_file = self.cache_dir / f"{key}.json"
        cache_data = {
            'value': value,
            'expires_at': datetime.now().timestamp() + ttl
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    async def cache_get(self, key: str) -> Optional[any]:
        """Get cache value"""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if expired
            if datetime.now().timestamp() > cache_data['expires_at']:
                cache_file.unlink()
                return None
            
            return cache_data['value']
        except (json.JSONDecodeError, KeyError):
            # Remove corrupted cache file
            cache_file.unlink()
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cache value"""
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            cache_file.unlink()
            return True
        
        return False
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache files"""
        cleaned = 0
        current_time = datetime.now().timestamp()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                if current_time > cache_data['expires_at']:
                    cache_file.unlink()
                    cleaned += 1
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                # Remove corrupted cache files
                try:
                    cache_file.unlink()
                    cleaned += 1
                except FileNotFoundError:
                    pass
        
        return cleaned