import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional, Tuple
from pathlib import Path
from scipy.spatial.distance import cosine
import torch
import uuid
import os

from app.core.config import settings
from app.services.voice_speechbrain import VoiceSpeechBrainService

# MediaPipe will be imported lazily to avoid startup issues
MEDIAPIPE_AVAILABLE = None  # Will be checked lazily

class BiometricService:
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.face_model_path = self.models_dir / settings.FACE_MODEL_PATH
        
        # ONNX Runtime sessions (lazy loading)
        self._face_session: Optional[ort.InferenceSession] = None
        
        # Face detection (lazy loading)
        self._face_detector = None
        
        # Voice service using SpeechBrain
        self.voice_service = VoiceSpeechBrainService()
        
        # Model specifications
        self.face_input_size = (160, 160)  # FaceNet standard
        
        # Debug settings
        self.debug_mode = settings.BIOMETRIC_DEBUG
        self.debug_dir = Path("debug_output")
        if self.debug_mode:
            self.debug_dir.mkdir(exist_ok=True)
            print(f"🐛 Debug mode enabled - saving debug images to {self.debug_dir}")
    
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
    
    def _detect_and_crop_face(self, image: Image.Image, debug_prefix: str = None) -> Image.Image:
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
            cropped = image.crop((left, top, right, bottom))
            
            if self.debug_mode and debug_prefix:
                self._save_debug_image(image, None, (left, top, right, bottom), f"{debug_prefix}_fallback")
            
            return cropped
        
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
                cropped = image.crop((left, top, right, bottom))
                
                if self.debug_mode and debug_prefix:
                    self._save_debug_image(image, None, (left, top, right, bottom), f"{debug_prefix}_no_face")
                
                return cropped
            
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
            
            # Store original detection box for debug
            original_box = (x, y, x + width, y + height)
            
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
            
            # Final crop box
            crop_box = (x, y, x + width, y + height)
            
            # Crop face from image
            face_image = image.crop(crop_box)
            
            confidence = detection.score[0] if detection.score else 0.0
            print(f"✅ Face detected with confidence: {confidence:.3f}")
            print(f"   Detection box: {original_box}")
            print(f"   Crop box (with margin): {crop_box}")
            
            # Save debug images if enabled
            if self.debug_mode and debug_prefix:
                self._save_debug_image(image, original_box, crop_box, f"{debug_prefix}_detected", confidence)
                # Also save the cropped face with unique ID
                unique_id = uuid.uuid4().hex[:8]
                cropped_filename = f"{debug_prefix}_cropped_{unique_id}.jpg"
                face_image.save(self.debug_dir / cropped_filename)
                print(f"🐛 Cropped face saved: {self.debug_dir / cropped_filename}")
            
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
            cropped = image.crop((left, top, right, bottom))
            
            if self.debug_mode and debug_prefix:
                self._save_debug_image(image, None, (left, top, right, bottom), f"{debug_prefix}_error")
            
            return cropped
    
    def _save_debug_image(self, original_image: Image.Image, detection_box: Optional[Tuple[int, int, int, int]], 
                         crop_box: Tuple[int, int, int, int], prefix: str, confidence: float = None):
        """Save debug image with bounding boxes drawn"""
        try:
            # Create a copy for drawing
            debug_image = original_image.copy()
            draw = ImageDraw.Draw(debug_image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("Arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw detection box in red if available
            if detection_box:
                draw.rectangle(detection_box, outline="red", width=3)
                if confidence:
                    draw.text((detection_box[0], detection_box[1] - 25), f"Detection: {confidence:.3f}", 
                             fill="red", font=font)
            
            # Draw crop box in green
            draw.rectangle(crop_box, outline="green", width=2)
            draw.text((crop_box[0], crop_box[1] - 25 if not detection_box else crop_box[1] + crop_box[3] + 5), 
                     "Crop area", fill="green", font=font)
            
            # Save debug image
            debug_filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
            debug_path = self.debug_dir / debug_filename
            debug_image.save(debug_path)
            print(f"🐛 Debug image saved: {debug_path}")
            
        except Exception as e:
            print(f"❌ Failed to save debug image: {e}")
    
    def _preprocess_face_image(self, image_data: bytes, debug_prefix: str = None) -> np.ndarray:
        """Preprocess face image for model inference"""
        try:
            # Load and convert image
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detect/crop face region
            face_image = self._detect_and_crop_face(image, debug_prefix)
            
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
    
    
    async def extract_face_embedding(self, image_data: bytes, debug_prefix: str = None) -> np.ndarray:
        """Extract face embedding from image"""
        try:
            preprocessed_image = self._preprocess_face_image(image_data, debug_prefix)
            
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
        """Extract voice embedding from audio using SpeechBrain"""
        import tempfile
        import soundfile as sf
        
        # First, try to determine the audio format and convert if needed
        temp_path = None
        temp_input_path = None
        
        try:
            # Save audio bytes to temporary file with original format
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_input_path = temp_file.name
            
            # Convert to WAV format for SpeechBrain processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                temp_path = wav_file.name
            
            # Use librosa to load and convert the audio
            import librosa
            audio, sr = librosa.load(temp_input_path, sr=16000, mono=True)
            
            # Save as WAV file
            sf.write(temp_path, audio, sr)
            
            # Clean up input file
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
                temp_input_path = None
            
            # Use SpeechBrain service to extract embedding
            embedding = self.voice_service.extract_voice_embedding(temp_path)
            return embedding
                
        finally:
            # Clean up temporary files
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
    
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
        face_model_available = self.face_model_path.exists()
        
        return {
            "face_model": {
                "loaded": face_model_available,  # Show available if file exists (lazy loading)
                "path": str(self.face_model_path),
                "exists": face_model_available,
                "input_size": self.face_input_size,
                "session_loaded": self._face_session is not None
            },
            "voice_model": {
                "loaded": self.voice_service.is_models_loaded(),
                "path": self.voice_service.model_path,
                "exists": Path(self.voice_service.model_path).exists(),
                "speechbrain_service": True
            },
            "face_detection": {
                "mediapipe_available": MEDIAPIPE_AVAILABLE if MEDIAPIPE_AVAILABLE is not None else "not_checked",
                "detector_loaded": self._face_detector is not None,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }