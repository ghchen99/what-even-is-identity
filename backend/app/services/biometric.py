import numpy as np
import onnxruntime as ort
from PIL import Image
import io
from typing import Optional, Tuple
from pathlib import Path
from scipy.spatial.distance import cosine
import torch

from app.core.config import settings

# MediaPipe will be imported lazily to avoid startup issues
MEDIAPIPE_AVAILABLE = None  # Will be checked lazily

class BiometricService:
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.face_model_path = self.models_dir / settings.FACE_MODEL_PATH
        self.voice_model_path = self.models_dir / settings.VOICE_MODEL_PATH
        
        # ONNX Runtime sessions (lazy loading)
        self._face_session: Optional[ort.InferenceSession] = None
        self._voice_session: Optional[ort.InferenceSession] = None
        
        # Face detection (lazy loading)
        self._face_detector = None
        
        # Model specifications
        self.face_input_size = (160, 160)  # FaceNet standard
        self.voice_sample_rate = 16000     # Standard sample rate
    
    def _load_face_model(self) -> Optional[ort.InferenceSession]:
        """Lazy load face recognition model"""
        if self._face_session is None:
            if not self.face_model_path.exists():
                print(f"Warning: Face model not found at {self.face_model_path}")
                return None
            
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self._face_session = ort.InferenceSession(
                str(self.face_model_path), 
                providers=providers
            )
        
        return self._face_session
    
    def _load_voice_model(self) -> Optional[ort.InferenceSession]:
        """Lazy load voice recognition model"""
        if self._voice_session is None:
            if not self.voice_model_path.exists():
                print(f"Warning: Voice model not found at {self.voice_model_path}")
                return None
            
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self._voice_session = ort.InferenceSession(
                str(self.voice_model_path), 
                providers=providers
            )
        
        return self._voice_session
    
    def _load_face_detector(self):
        """Lazy load MediaPipe face detector"""
        global MEDIAPIPE_AVAILABLE
        
        # Check MediaPipe availability lazily
        if MEDIAPIPE_AVAILABLE is None:
            try:
                import mediapipe as mp
                MEDIAPIPE_AVAILABLE = True
                print("✅ MediaPipe available")
            except ImportError:
                MEDIAPIPE_AVAILABLE = False
                print("⚠️  MediaPipe not available. Install with: pip install mediapipe")
                return None
        
        if not MEDIAPIPE_AVAILABLE:
            return None
            
        if self._face_detector is None:
            try:
                print("🔄 Loading MediaPipe face detector...")
                import mediapipe as mp
                # Initialize MediaPipe Face Detection
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,  # 0 for short-range (2m), 1 for full-range (5m)
                    min_detection_confidence=0.5
                )
                print("✅ MediaPipe face detector loaded")
            except Exception as e:
                print(f"❌ Failed to load MediaPipe: {e}")
                return None
        
        return self._face_detector
    
    def _detect_and_crop_face(self, image: Image.Image) -> Image.Image:
        """Detect and crop face from image using MediaPipe"""
        detector = self._load_face_detector()
        
        if detector is None:
            print("⚠️  MediaPipe not available, using center crop fallback")
            # Fallback to center crop
            width, height = image.size
            crop_size = min(width, height) * 0.7
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = left + crop_size
            bottom = top + crop_size
            return image.crop((left, top, right, bottom))
        
        try:
            # Convert PIL to RGB numpy array for MediaPipe
            img_rgb = np.array(image.convert('RGB'))
            
            # Detect faces
            results = detector.process(img_rgb)
            
            if not results.detections:
                print("⚠️  No face detected, using center crop fallback")
                # Fallback to center crop
                width, height = image.size
                crop_size = min(width, height) * 0.7
                left = (width - crop_size) / 2
                top = (height - crop_size) / 2  
                right = left + crop_size
                bottom = top + crop_size
                return image.crop((left, top, right, bottom))
            
            # Use the first face detection (highest confidence by default)
            detection = results.detections[0]
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Extract bounding box (MediaPipe returns relative coordinates)
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x = int(bbox.xmin * img_width)
            y = int(bbox.ymin * img_height)
            width = int(bbox.width * img_width)
            height = int(bbox.height * img_height)
            
            # Add margin around face
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            width = width + 2 * margin
            height = height + 2 * margin
            
            # Ensure we don't go outside image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            
            # Crop face from image
            face_image = image.crop((x, y, x + width, y + height))
            
            confidence = detection.score[0] if detection.score else 0.0
            print(f"✅ Face detected with confidence: {confidence:.3f}")
            return face_image
            
        except Exception as e:
            print(f"❌ Face detection failed: {e}, using center crop fallback")
            # Fallback to center crop
            width, height = image.size
            crop_size = min(width, height) * 0.7
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = left + crop_size
            bottom = top + crop_size
            return image.crop((left, top, right, bottom))
    
    def _preprocess_face_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess face image for model inference"""
        try:
            # Load and convert image
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detect/crop face region
            face_image = self._detect_and_crop_face(image)
            
            # Resize to model input size
            face_image = face_image.resize(self.face_input_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy and normalize
            image_array = np.array(face_image, dtype=np.float32)
            
            # FaceNet normalization: [-1, 1]  
            image_array = (image_array - 127.5) / 128.0
            
            # Rearrange to NCHW format
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
            image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess face image: {str(e)}")
    
    def _preprocess_voice_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess voice audio for model inference"""
        try:
            # Mock audio preprocessing
            # In production, this would:
            # 1. Decode audio bytes
            # 2. Resample to target rate
            # 3. Extract features (MFCC, spectrogram)
            # 4. Format for model input
            
            # Mock feature vector
            mock_features = np.random.normal(0, 1, (1, 80, 300)).astype(np.float32)
            return mock_features
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess voice audio: {str(e)}")
    
    async def extract_face_embedding(self, image_data: bytes) -> np.ndarray:
        """Extract face embedding from image"""
        try:
            preprocessed_image = self._preprocess_face_image(image_data)
            
            session = self._load_face_model()
            
            if session is None:
                # Mock embedding when model not available
                print("Using mock face embedding (model not loaded)")
                embedding = np.random.normal(0, 1, 512).astype(np.float32)
                return embedding / np.linalg.norm(embedding)
            
            # Run inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            result = session.run([output_name], {input_name: preprocessed_image})
            embedding = result[0].flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Face embedding extraction failed: {str(e)}")
            # Fallback to mock embedding
            embedding = np.random.normal(0, 1, 512).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
    
    async def extract_voice_embedding(self, audio_data: bytes) -> np.ndarray:
        """Extract voice embedding from audio"""
        try:
            preprocessed_audio = self._preprocess_voice_audio(audio_data)
            
            session = self._load_voice_model()
            
            if session is None:
                # Mock embedding when model not available
                print("Using mock voice embedding (model not loaded)")
                embedding = np.random.normal(0, 1, 256).astype(np.float32)
                return embedding / np.linalg.norm(embedding)
            
            # Run inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            result = session.run([output_name], {input_name: preprocessed_audio})
            embedding = result[0].flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Voice embedding extraction failed: {str(e)}")
            # Fallback to mock embedding
            embedding = np.random.normal(0, 1, 256).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Clamp to [0, 1] range
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            print(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    def calculate_combined_score(self, face_similarity: float, voice_similarity: float) -> float:
        """Calculate weighted combined similarity score"""
        return (face_similarity * settings.FACE_WEIGHT) + (voice_similarity * settings.VOICE_WEIGHT)
    
    def validate_image_format(self, image_data: bytes) -> bool:
        """Validate uploaded image format"""
        try:
            image = Image.open(io.BytesIO(image_data))
            return image.format in ['JPEG', 'PNG', 'BMP', 'WEBP']
        except Exception:
            return False
    
    def validate_audio_format(self, audio_data: bytes) -> bool:
        """Validate uploaded audio format"""
        try:
            # Basic size check
            return len(audio_data) > 1000  # At least 1KB
        except Exception:
            return False
    
    def get_model_info(self) -> dict:
        """Get model loading status and information"""
        return {
            "face_model": {
                "loaded": self._face_session is not None,
                "path": str(self.face_model_path),
                "exists": self.face_model_path.exists(),
                "input_size": self.face_input_size
            },
            "voice_model": {
                "loaded": self._voice_session is not None,
                "path": str(self.voice_model_path),
                "exists": self.voice_model_path.exists(),
                "sample_rate": self.voice_sample_rate
            },
            "face_detection": {
                "mediapipe_available": MEDIAPIPE_AVAILABLE if MEDIAPIPE_AVAILABLE is not None else "not_checked",
                "detector_loaded": self._face_detector is not None,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }