#!/usr/bin/env python3
"""
API Testing Script for Biometric Unlock System

This script tests all API endpoints with sample data and demonstrates
the complete enrollment and verification workflow.

Usage:
1. Start the backend server: cd backend && python run.py
2. Run this test script: python test_api.py
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TEST_USER_ID = "test-user-123"
TEST_USER_DIR = Path("tests/fixtures")

class APITester:
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.enrollment_id = None
        self.verification_id = None
        
    def get_available_files(self):
        """Get available test files from test-user directory"""
        images_dir = TEST_USER_DIR / "images"
        voice_dir = TEST_USER_DIR / "voice"
        
        # Get available files
        image_files = []
        voice_files = []
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if voice_dir.exists():
            voice_files = list(voice_dir.glob("*.mp3")) + list(voice_dir.glob("*.wav")) + list(voice_dir.glob("*.m4a"))
        
        if not image_files or not voice_files:
            raise FileNotFoundError(
                f"Missing test files! Please ensure you have:\n"
                f"  - Images in: {images_dir}/\n" 
                f"  - Voice files in: {voice_dir}/\n"
                f"Found: {len(image_files)} images, {len(voice_files)} voice files"
            )
        
        return sorted(image_files), sorted(voice_files)
    
    def setup_test_data(self):
        """Setup test data using uploaded files"""
        print("📁 Setting up test data from uploaded files...")
        
        try:
            # Get available files
            image_files, voice_files = self.get_available_files()
            
            print(f"✅ Found test files:")
            print(f"   - Images: {len(image_files)} files")
            print(f"   - Voice: {len(voice_files)} files")
            
            # Use first files for enrollment
            self.enrollment_face = image_files[0]
            self.enrollment_voice = voice_files[0]
            
            # Use second files for successful verification (if available)
            if len(image_files) > 1 and len(voice_files) > 1:
                self.verification_face = image_files[1]
                self.verification_voice = voice_files[1]
            else:
                # Use same files if only one available
                self.verification_face = image_files[0]
                self.verification_voice = voice_files[0]
            
            # Use different files for failed verification (if available)
            if len(image_files) > 2 and len(voice_files) > 2:
                self.different_face = image_files[2]
                self.different_voice = voice_files[2]
            else:
                # Use last available files
                self.different_face = image_files[-1]
                self.different_voice = voice_files[-1]
            
            print(f"📝 Test file assignments:")
            print(f"   - Enrollment: {self.enrollment_face.name} + {self.enrollment_voice.name}")
            print(f"   - Verification: {self.verification_face.name} + {self.verification_voice.name}")
            print(f"   - Different user: {self.different_face.name} + {self.different_voice.name}")
            print()
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            raise
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   - Face model loaded: {data['models']['face_model']['loaded']}")
            print(f"   - Voice model loaded: {data['models']['voice_model']['loaded']}")
            print(f"   - MediaPipe available: {data['models']['face_detection']['mediapipe_available']}")
            print()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_enrollment(self):
        """Test user enrollment"""
        print(f"📝 Testing enrollment for user: {TEST_USER_ID}")
        
        try:
            with open(self.enrollment_face, 'rb') as face_file, \
                 open(self.enrollment_voice, 'rb') as voice_file:
                
                files = {
                    'face_image': ('face.jpg', face_file, 'image/jpeg'),
                    'voice_audio': ('voice.mp3', voice_file, 'audio/mp3')
                }
                
                data = {
                    'user_id': TEST_USER_ID
                }
                
                response = self.session.post(f"{self.base_url}/biometric/enroll", 
                                           files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                self.enrollment_id = result['enrollment_id']
                
                print(f"✅ Enrollment successful!")
                print(f"   - User ID: {result['user_id']}")
                print(f"   - Enrollment ID: {result['enrollment_id']}")
                print(f"   - Status: {result['status']}")
                print()
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Enrollment failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Error details: {e.response.text}")
            return False
    
    def test_verification(self, face_file, voice_file, expected_result=True, threshold=0.6):
        """Test user verification"""
        print(f"🔍 Testing verification (expected: {'PASS' if expected_result else 'FAIL'})...")
        
        try:
            with open(face_file, 'rb') as face, \
                 open(voice_file, 'rb') as voice:
                
                files = {
                    'face_image': ('face.jpg', face, 'image/jpeg'),
                    'voice_audio': ('voice.mp3', voice, 'audio/mp3')
                }
                
                data = {
                    'user_id': TEST_USER_ID,
                    'threshold': threshold
                }
                
                response = self.session.post(f"{self.base_url}/biometric/verify", 
                                           files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                self.verification_id = result['verification_id']
                
                print(f"✅ Verification completed!")
                print(f"   - Verified: {result['verified']}")
                print(f"   - Face similarity: {result['face_similarity']:.3f}")
                print(f"   - Voice similarity: {result['voice_similarity']:.3f}")
                print(f"   - Combined score: {result['combined_score']:.3f}")
                print(f"   - Threshold: {result['threshold']}")
                print(f"   - Verification ID: {result['verification_id']}")
                print()
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Verification failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Error details: {e.response.text}")
            return False
    
    def test_user_enrollments(self):
        """Test getting user enrollments"""
        print(f"📋 Testing get user enrollments...")
        
        try:
            response = self.session.get(f"{self.base_url}/users/{TEST_USER_ID}/enrollments")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Retrieved {len(data['enrollments'])} enrollments")
            
            for i, enrollment in enumerate(data['enrollments']):
                print(f"   Enrollment {i+1}:")
                print(f"     - ID: {enrollment['enrollment_id']}")
                print(f"     - Date: {enrollment['enrolled_at']}")
                print(f"     - Face embedding size: {enrollment['face_embedding_size']}")
                print(f"     - Voice embedding size: {enrollment['voice_embedding_size']}")
            print()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Get enrollments failed: {e}")
            return False
    
    def test_user_verifications(self):
        """Test getting user verifications"""
        print(f"📊 Testing get user verifications...")
        
        try:
            response = self.session.get(f"{self.base_url}/users/{TEST_USER_ID}/verifications?limit=5")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Retrieved {len(data['verifications'])} verifications")
            
            for i, verification in enumerate(data['verifications']):
                print(f"   Verification {i+1}:")
                print(f"     - ID: {verification['verification_id']}")
                print(f"     - Date: {verification['timestamp']}")
                print(f"     - Verified: {verification['verified']}")
                print(f"     - Combined score: {verification['combined_score']:.3f}")
            print()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Get verifications failed: {e}")
            return False
    
    def test_cache_cleanup(self):
        """Test cache cleanup"""
        print("🧹 Testing cache cleanup...")
        
        try:
            response = self.session.post(f"{self.base_url}/health/cache/cleanup")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Cache cleanup completed")
            print(f"   - Cleaned entries: {data['cleaned_entries']}")
            print(f"   - Status: {data['status']}")
            print()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Cache cleanup failed: {e}")
            return False
    
    def test_delete_user(self):
        """Test deleting user data"""
        print(f"🗑️  Testing delete user: {TEST_USER_ID}")
        
        try:
            response = self.session.delete(f"{self.base_url}/users/{TEST_USER_ID}")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ User deleted successfully")
            print(f"   - Status: {data['status']}")
            print(f"   - Message: {data['message']}")
            print()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Delete user failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Biometric API Test Suite")
        print("=" * 50)
        
        # Setup test data
        self.setup_test_data()
        
        # Test sequence
        tests = [
            ("Health Check", self.test_health_check),
            ("User Enrollment", self.test_enrollment),
            ("Same User Verification", lambda: self.test_verification(
                self.verification_face, self.verification_voice, True, 0.6)),
            ("Different User Verification", lambda: self.test_verification(
                self.different_face, self.different_voice, False, 0.6)),
            ("High Threshold Verification", lambda: self.test_verification(
                self.verification_face, self.verification_voice, True, 0.95)),
            ("Get User Enrollments", self.test_user_enrollments),
            ("Get User Verifications", self.test_user_verifications),
            ("Cache Cleanup", self.test_cache_cleanup),
            ("Delete User", self.test_delete_user),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"🧪 Running: {test_name}")
            if test_func():
                passed += 1
            else:
                print(f"⚠️  {test_name} failed, continuing...")
            
            time.sleep(1)  # Brief pause between tests
        
        print("=" * 50)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {total - passed} tests failed")
        
        return passed == total

def main():
    """Main function"""
    print("Biometric API Testing Script")
    print("Make sure the backend server is running on http://localhost:8000")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Is it running?")
        print("Start it with: cd backend && python run.py")
        return
    
    print("✅ Server is running, starting tests...")
    print()
    
    # Run tests
    tester = APITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎊 All tests completed successfully!")
        print(f"📁 Test data saved in: {TEST_USER_DIR}")
        print("💡 You can examine the generated test files and backend/data/ folder")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()