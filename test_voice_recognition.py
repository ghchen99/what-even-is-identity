#!/usr/bin/env python3
"""
Simple X-Vector Voice Recognition Test Script
This script tests the x-vector model with minimal preprocessing to understand what it actually expects.
"""

import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from pathlib import Path
from scipy.spatial.distance import cosine
import sys

def load_audio_simple(audio_path, target_sr=16000):
    """Load audio file with minimal processing"""
    print(f"Loading: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    print(f"Original: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f}s)")
    
    return audio, sr

def preprocess_for_xvector_v1(audio, sr=16000):
    """Version 1: Minimal preprocessing - just MFCC features"""
    # Extract MFCC features (common for x-vector models)
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=13,  # Standard number of MFCC coefficients
        n_fft=512,
        hop_length=160,
        win_length=400
    )
    
    # Transpose to time x features
    features = mfccs.T
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0).astype(np.float32)
    
    print(f"MFCC features shape: {features.shape}")
    return features

def preprocess_for_xvector_v2(audio, sr=16000):
    """Version 2: Log-mel spectrogram (what we're currently using)"""
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=sr // 2,
        power=2.0
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to time x features format
    log_mel_spec = log_mel_spec.T
    
    # Normalize
    mean = np.mean(log_mel_spec, axis=0, keepdims=True)
    std = np.std(log_mel_spec, axis=0, keepdims=True)
    log_mel_spec = (log_mel_spec - mean) / (std + 1e-8)
    
    # Add batch dimension
    features = np.expand_dims(log_mel_spec, axis=0).astype(np.float32)
    
    print(f"Log-mel features shape: {features.shape}")
    return features

def preprocess_for_xvector_v3(audio, sr=16000):
    """Version 3: Raw spectrogram"""
    # Simple spectrogram
    stft = librosa.stft(audio, n_fft=512, hop_length=160, win_length=400)
    magnitude = np.abs(stft)
    
    # Transpose to time x features
    features = magnitude.T
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0).astype(np.float32)
    
    print(f"Spectrogram features shape: {features.shape}")
    return features

def run_xvector_model(features, model_path):
    """Run the x-vector model and return embedding"""
    try:
        # Load model
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"Model expects input '{input_name}' with shape: {input_shape}")
        print(f"Our features shape: {features.shape}")
        
        # Run inference
        result = session.run([output_name], {input_name: features})
        embedding = result[0]
        
        print(f"Raw embedding shape: {embedding.shape}")
        print(f"Raw embedding stats: min={np.min(embedding):.6f}, max={np.max(embedding):.6f}, mean={np.mean(embedding):.6f}")
        
        # Flatten and normalize
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    except Exception as e:
        print(f"Model inference failed: {e}")
        return None

def calculate_similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    if emb1 is None or emb2 is None:
        return 0.0
    
    similarity = 1 - cosine(emb1, emb2)
    return similarity

def create_synthetic_voice(duration=3.0, sr=16000, freq_base=200):
    """Create synthetic voice-like audio for testing"""
    t = np.linspace(0, duration, int(duration * sr))
    
    # Create a voice-like signal with formants
    signal = np.zeros_like(t)
    
    # Add multiple harmonics to simulate voice
    for harmonic in range(1, 8):
        freq = freq_base * harmonic
        if freq < sr / 2:  # Below Nyquist frequency
            amplitude = 1.0 / harmonic  # Decreasing amplitude
            signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    signal += 0.1 * np.random.randn(len(t))
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

def main():
    print("🎤 X-Vector Voice Recognition Test")
    print("=" * 50)
    
    # Paths
    model_path = Path("backend/models/xvector.onnx")
    audio_dir = Path("backend/tests/fixtures/voice")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("📁 Testing with real audio files...")
    real_audio_files = []
    if audio_dir.exists():
        real_audio_files = list(audio_dir.glob("*.mp3"))
        print(f"Found {len(real_audio_files)} real audio files")
    
    print("🎭 Creating synthetic test voices...")
    synthetic_voices = [
        ("synthetic_voice_low", create_synthetic_voice(freq_base=150)),   # Lower pitch
        ("synthetic_voice_mid", create_synthetic_voice(freq_base=200)),   # Medium pitch  
        ("synthetic_voice_high", create_synthetic_voice(freq_base=300)),  # Higher pitch
    ]
    
    all_audio_sources = []
    
    # Add real files
    for audio_file in real_audio_files[:3]:
        all_audio_sources.append(("real", audio_file, None))
    
    # Add synthetic voices
    for name, audio_data in synthetic_voices:
        all_audio_sources.append(("synthetic", name, audio_data))
    
    print(f"Total audio sources: {len(all_audio_sources)}")
    
    # Test only the working method (Log-Mel)
    print(f"\n🧪 Testing Log-Mel Preprocessing")
    print("-" * 50)
    
    embeddings = []
    
    # Process each audio source
    for source_type, source_name, audio_data in all_audio_sources:
        print(f"\n🎵 Processing: {source_name} ({source_type})")
        
        try:
            if source_type == "real":
                # Load real audio file
                audio, sr = load_audio_simple(source_name)
            else:
                # Use synthetic audio
                audio = audio_data
                sr = 16000
                print(f"Synthetic: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f}s)")
            
            # Preprocess with log-mel
            features = preprocess_for_xvector_v2(audio, sr)
            
            # Run model
            embedding = run_xvector_model(features, model_path)
            
            if embedding is not None:
                embeddings.append((str(source_name), embedding))
                print(f"✅ Generated embedding: {embedding.shape}")
            else:
                print("❌ Failed to generate embedding")
                
        except Exception as e:
            print(f"❌ Error processing {source_name}: {e}")
    
    # Calculate similarities
    if len(embeddings) >= 2:
        print(f"\n📊 Voice Similarity Matrix:")
        print("-" * 60)
        
        # Print header
        names = [name for name, _ in embeddings]
        print(f"{'':20}", end="")
        for name in names:
            display_name = Path(name).stem if "/" in name else name
            print(f"{display_name[:15]:>15}", end="")
        print()
        
        # Print similarity matrix
        for i, (name1, emb1) in enumerate(embeddings):
            display_name1 = Path(name1).stem if "/" in name1 else name1
            print(f"{display_name1[:20]:20}", end="")
            for j, (name2, emb2) in enumerate(embeddings):
                if i == j:
                    print(f"{'1.000':>15}", end="")
                else:
                    similarity = calculate_similarity(emb1, emb2)
                    print(f"{similarity:>15.6f}", end="")
            print()
            
        print(f"\n🎯 Key Findings:")
        print("- Values close to 1.0 = Same speaker")  
        print("- Values below 0.8 = Different speakers")
        print("- Real audio files are all from user001 (same person)")
        print("- Synthetic voices should show lower similarity to real voices")

if __name__ == "__main__":
    main()