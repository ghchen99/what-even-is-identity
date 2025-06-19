"""
SpeechBrain ECAPA-TDNN Voice Recognition Service
Using the official SpeechBrain implementation instead of ONNX
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import logging
from typing import Optional, Tuple

# Suppress some verbose logging
logging.getLogger('speechbrain').setLevel(logging.WARNING)


class VoiceSpeechBrainService:
    """Voice recognition service using SpeechBrain ECAPA-TDNN models"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Use absolute path from current working directory
            from pathlib import Path
            cwd = Path.cwd()
            # Check if we're in the backend directory or need to go up
            if cwd.name == "backend":
                self.model_path = str(cwd / "models" / "spkrec-ecapa-voxceleb")
            else:
                self.model_path = str(cwd / "backend" / "models" / "spkrec-ecapa-voxceleb")
        else:
            self.model_path = model_path
        self.encoder = None
        self.verification = None
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load SpeechBrain ECAPA-TDNN models from local filepath"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier, SpeakerRecognition
            
            # Check if model directory exists
            model_path = Path(self.model_path)
            if not model_path.exists():
                logging.warning(f"Model path does not exist: {self.model_path}")
                logging.info("Attempting to download models from HuggingFace...")
                # Try to download from HuggingFace
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=self.model_path
                )
                
                self.verification = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", 
                    savedir=self.model_path
                )
                logging.info("Models downloaded and loaded successfully")
                return True
            
            logging.info("Loading ECAPA-TDNN encoder from local path...")
            self.encoder = EncoderClassifier.from_hparams(
                source=self.model_path,
                savedir=self.model_path
            )
            
            logging.info("Loading speaker verification model from local path...")
            self.verification = SpeakerRecognition.from_hparams(
                source=self.model_path, 
                savedir=self.model_path
            )
            
            logging.info("Models loaded successfully from local path")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            return False
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> Optional[Tuple[torch.Tensor, int]]:
        """Load audio file for SpeechBrain processing"""
        try:
            # Load with torchaudio (preferred by SpeechBrain)
            signal, fs = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if fs != target_sr:
                resampler = torchaudio.transforms.Resample(fs, target_sr)
                signal = resampler(signal)
                fs = target_sr
            
            # Convert to mono if stereo
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            return signal, fs
            
        except Exception as e:
            logging.error(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_voice_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract voice embedding from audio file"""
        if self.encoder is None:
            logging.error("Encoder model not loaded")
            raise RuntimeError("Voice encoder model not available")
        
        try:
            signal, fs = self.load_audio(audio_path)
            if signal is None:
                raise ValueError(f"Failed to load audio from {audio_path}")
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(signal)
                # Convert to numpy and normalize
                embedding_np = embedding.squeeze().cpu().numpy()
                embedding_np = embedding_np / np.linalg.norm(embedding_np)
                
            return embedding_np
            
        except Exception as e:
            logging.error(f"Error extracting embedding from {audio_path}: {e}")
            raise
    
    def verify_speaker(self, audio_path1: str, audio_path2: str) -> Optional[Tuple[float, bool]]:
        """Verify if two audio files are from the same speaker"""
        if self.verification is None:
            logging.error("Verification model not loaded")
            raise RuntimeError("Voice verification model not available")
        
        try:
            score, prediction = self.verification.verify_files(audio_path1, audio_path2)
            return score[0].item(), prediction[0].item()
            
        except Exception as e:
            logging.error(f"Error verifying speakers: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            return 1 - cosine(embedding1, embedding2)
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def is_models_loaded(self) -> bool:
        """Check if models are loaded successfully"""
        return self.encoder is not None and self.verification is not None