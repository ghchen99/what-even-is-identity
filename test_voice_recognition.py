#!/usr/bin/env python3
"""
ECAPA-TDNN Voice Recognition Test Script
This script tests the ECAPA-TDNN speaker model with proper MFCC preprocessing.
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

def preprocess_for_ecapa_tdnn(audio, sr=16000):
    """ECAPA-TDNN preprocessing - 80 MFCC features"""
    # Extract MFCC features (80 features for ECAPA-TDNN)
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=80,  # ECAPA-TDNN expects 80 MFCC coefficients
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


def run_ecapa_tdnn_model(features, model_path):
    """Run the ECAPA-TDNN model and return embedding"""
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
    print("🎤 ECAPA-TDNN Voice Recognition Test")
    print("=" * 50)
    
    # Paths
    model_path = Path("backend/models/ecapa_tdnn.onnx")
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
    
    # Test ECAPA-TDNN preprocessing
    print(f"\n🧪 Testing ECAPA-TDNN MFCC Preprocessing")
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
            
            # Preprocess with MFCC for ECAPA-TDNN
            features = preprocess_for_ecapa_tdnn(audio, sr)
            
            # Run model
            embedding = run_ecapa_tdnn_model(features, model_path)
            
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
        print("- Values ≥95% = Same speaker ✅")  
        print("- Values 60-80% = Different speakers (acceptable)")
        print("- Values ≤60% = Very different speakers/synthetic ✅")
        print("- ECAPA-TDNN threshold recommendations:")
        print("  • Same speaker verification: ≥95%")
        print("  • Different speaker rejection: ≤75%")
        
        # Analyze results
        print(f"\n📈 Analysis:")
        same_speaker_pairs = []
        different_speaker_pairs = []
        
        for i, (name1, emb1) in enumerate(embeddings):
            for j, (name2, emb2) in enumerate(embeddings):
                if i < j:  # Avoid duplicates
                    similarity = calculate_similarity(emb1, emb2)
                    name1_clean = Path(name1).stem if "/" in name1 else name1
                    name2_clean = Path(name2).stem if "/" in name2 else name2
                    
                    # Check if same speaker (user001 files)
                    if "user001" in name1_clean and "user001" in name2_clean:
                        same_speaker_pairs.append((name1_clean, name2_clean, similarity))
                    elif "synthetic" not in name1_clean and "synthetic" not in name2_clean:
                        different_speaker_pairs.append((name1_clean, name2_clean, similarity))
        
        if same_speaker_pairs:
            print("Same Speaker Comparisons:")
            for n1, n2, sim in same_speaker_pairs:
                status = "✅ PASS" if sim >= 0.95 else "❌ FAIL"
                print(f"  {n1} vs {n2}: {sim:.3f} ({sim*100:.1f}%) {status}")
        
        if different_speaker_pairs:
            print("Different Speaker Comparisons:")
            for n1, n2, sim in different_speaker_pairs:
                status = "✅ PASS" if sim <= 0.80 else "⚠️  HIGH" 
                print(f"  {n1} vs {n2}: {sim:.3f} ({sim*100:.1f}%) {status}")
                if sim > 0.80:
                    print(f"    Note: High similarity might indicate similar speakers or recording conditions")

if __name__ == "__main__":
    main()