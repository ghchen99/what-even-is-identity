#!/usr/bin/env python3
"""
SpeechBrain ECAPA-TDNN Voice Recognition Script
Using the official SpeechBrain implementation instead of ONNX
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import sys
import logging

# Suppress some verbose logging
logging.getLogger('speechbrain').setLevel(logging.WARNING)

def install_requirements():
    """Check and install required packages"""
    try:
        import speechbrain
        print("✅ SpeechBrain already installed")
    except ImportError:
        print("📦 Installing SpeechBrain...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/speechbrain/speechbrain.git@develop"
        ])
        print("✅ SpeechBrain installed successfully")

def load_models():
    """Load SpeechBrain ECAPA-TDNN models"""
    try:
        from speechbrain.inference.speaker import EncoderClassifier, SpeakerRecognition
        
        print("🔄 Loading ECAPA-TDNN encoder...")
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        
        print("🔄 Loading speaker verification model...")
        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        
        print("✅ Models loaded successfully")
        return encoder, verification
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return None, None

def load_audio_for_speechbrain(audio_path, target_sr=16000):
    """Load audio file for SpeechBrain processing"""
    print(f"Loading: {audio_path}")
    
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
        
        print(f"Loaded: {signal.shape} at {fs}Hz ({signal.shape[1]/fs:.2f}s)")
        return signal, fs
        
    except Exception as e:
        print(f"❌ Error loading {audio_path}: {e}")
        return None, None

def create_synthetic_audio_torch(duration=3.0, sr=16000, freq_base=200):
    """Create synthetic voice-like audio as PyTorch tensor"""
    t = torch.linspace(0, duration, int(duration * sr))
    
    # Create voice-like signal with harmonics
    signal = torch.zeros_like(t)
    
    for harmonic in range(1, 8):
        freq = freq_base * harmonic
        if freq < sr / 2:
            amplitude = 1.0 / harmonic
            signal += amplitude * torch.sin(2 * torch.pi * freq * t)
    
    # Add noise
    signal += 0.1 * torch.randn_like(t)
    
    # Normalize and add batch dimension
    signal = signal / torch.max(torch.abs(signal)) * 0.8
    signal = signal.unsqueeze(0)  # Add channel dimension
    
    return signal

def test_speechbrain_ecapa():
    """Test SpeechBrain ECAPA-TDNN implementation"""
    print("🎤 SpeechBrain ECAPA-TDNN Voice Recognition Test")
    print("=" * 60)
    
    # Install requirements if needed
    install_requirements()
    
    # Load models
    encoder, verification = load_models()
    if encoder is None or verification is None:
        return
    
    # Test data preparation
    audio_dir = Path("backend/tests/fixtures/voice")
    test_files = []
    
    # Real audio files
    if audio_dir.exists():
        real_files = list(audio_dir.glob("*.mp3"))[:4]  # Limit to 4 files
        test_files.extend(real_files)
        print(f"📁 Found {len(real_files)} real audio files")
    
    # Synthetic audio
    synthetic_voices = [
        ("synthetic_low", create_synthetic_audio_torch(freq_base=150)),
        ("synthetic_mid", create_synthetic_audio_torch(freq_base=200)), 
        ("synthetic_high", create_synthetic_audio_torch(freq_base=300)),
    ]
    
    # Silent audio
    silent_audio = torch.zeros(1, 48000)  # 3 seconds of silence
    synthetic_voices.append(("silent", silent_audio))
    
    print(f"🎭 Created {len(synthetic_voices)} synthetic test signals")
    
    # Test 1: Embedding Generation
    print(f"\n🧪 Test 1: Embedding Generation")
    print("-" * 40)
    
    embeddings = []
    
    # Process real files
    for audio_file in test_files:
        try:
            signal, fs = load_audio_for_speechbrain(audio_file)
            if signal is not None:
                # Generate embedding
                with torch.no_grad():
                    embedding = encoder.encode_batch(signal)
                    # Convert to numpy and normalize
                    embedding_np = embedding.squeeze().cpu().numpy()
                    embedding_np = embedding_np / np.linalg.norm(embedding_np)
                    
                embeddings.append((str(audio_file.name), embedding_np))
                print(f"✅ {audio_file.name}: shape {embedding_np.shape}")
        except Exception as e:
            print(f"❌ {audio_file.name}: {e}")
    
    # Process synthetic audio
    for name, signal in synthetic_voices:
        try:
            with torch.no_grad():
                embedding = encoder.encode_batch(signal)
                embedding_np = embedding.squeeze().cpu().numpy()
                embedding_np = embedding_np / np.linalg.norm(embedding_np)
                
            embeddings.append((name, embedding_np))
            print(f"✅ {name}: shape {embedding_np.shape}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    # Test 2: Similarity Matrix
    if len(embeddings) >= 2:
        print(f"\n📊 Test 2: Voice Similarity Matrix")
        print("-" * 40)
        
        # Print header
        names = [name for name, _ in embeddings]
        print(f"{'':20}", end="")
        for name in names:
            display_name = Path(name).stem if Path(name).suffix else name
            print(f"{display_name[:12]:>12}", end="")
        print()
        
        # Print similarity matrix
        similarities = []
        for i, (name1, emb1) in enumerate(embeddings):
            display_name1 = Path(name1).stem if Path(name1).suffix else name1
            print(f"{display_name1[:20]:20}", end="")
            
            row_similarities = []
            for j, (name2, emb2) in enumerate(embeddings):
                if i == j:
                    similarity = 1.0
                    print(f"{'1.000':>12}", end="")
                else:
                    similarity = 1 - cosine(emb1, emb2)
                    print(f"{similarity:>12.3f}", end="")
                row_similarities.append((name2, similarity))
            similarities.append((name1, row_similarities))
            print()
    
    # Test 3: Direct Verification (if we have real files)
    real_files = [f for f in test_files if f.suffix in ['.mp3', '.wav']]
    if len(real_files) >= 2:
        print(f"\n🎯 Test 3: Direct Speaker Verification")
        print("-" * 40)
        
        # Test same speaker (if available)
        same_speaker_files = []
        different_speaker_files = []
        
        for i, file1 in enumerate(real_files):
            for file2 in real_files[i+1:]:
                # Check if same speaker (assuming naming convention like user001-1.mp3)
                speaker1 = file1.stem.split('-')[0] if '-' in file1.stem else file1.stem
                speaker2 = file2.stem.split('-')[0] if '-' in file2.stem else file2.stem
                
                if speaker1 == speaker2:
                    same_speaker_files.append((file1, file2))
                else:
                    different_speaker_files.append((file1, file2))
        
        # Test same speaker pairs
        if same_speaker_files:
            print("Same Speaker Tests:")
            for file1, file2 in same_speaker_files[:3]:  # Limit to 3 tests
                try:
                    score, prediction = verification.verify_files(str(file1), str(file2))
                    status = "✅ SAME" if prediction[0].item() else "❌ DIFFERENT"
                    print(f"  {file1.name} vs {file2.name}")
                    print(f"    Score: {score[0].item():.6f}, Prediction: {status}")
                except Exception as e:
                    print(f"  ❌ Error testing {file1.name} vs {file2.name}: {e}")
        
        # Test different speaker pairs  
        if different_speaker_files:
            print("Different Speaker Tests:")
            for file1, file2 in different_speaker_files[:3]:  # Limit to 3 tests
                try:
                    score, prediction = verification.verify_files(str(file1), str(file2))
                    status = "✅ DIFFERENT" if not prediction[0].item() else "⚠️ SAME"
                    print(f"  {file1.name} vs {file2.name}")
                    print(f"    Score: {score[0].item():.6f}, Prediction: {status}")
                except Exception as e:
                    print(f"  ❌ Error testing {file1.name} vs {file2.name}: {e}")
    
    # Test 4: Silent Audio Analysis
    print(f"\n🔇 Test 4: Silent Audio Analysis")
    print("-" * 40)
    
    silent_embedding = None
    for name, emb in embeddings:
        if name == "silent":
            silent_embedding = emb
            break
    
    if silent_embedding is not None:
        print("Silent audio embedding stats:")
        print(f"  Mean: {np.mean(silent_embedding):.6f}")
        print(f"  Std: {np.std(silent_embedding):.6f}")
        print(f"  Norm: {np.linalg.norm(silent_embedding):.6f}")
        
        print("Silent vs other signals:")
        for name, emb in embeddings:
            if name != "silent":
                sim = 1 - cosine(silent_embedding, emb)
                status = "⚠️ HIGH" if sim > 0.3 else "✅ LOW"
                print(f"  Silent vs {name}: {sim:.6f} {status}")
    
    # Final Analysis
    print(f"\n📈 Final Analysis")
    print("-" * 40)
    print("✅ Expected Results:")
    print("  • Same speaker: Score > 0.5, Prediction = SAME")
    print("  • Different speakers: Score < 0.5, Prediction = DIFFERENT") 
    print("  • Silent audio: Very low similarity with voice (<0.3)")
    print("  • Synthetic audio: Low-medium similarity with real voice")
    
    print("\n💡 Advantages of SpeechBrain over ONNX:")
    print("  • Automatic preprocessing and normalization")
    print("  • Built-in speaker verification pipeline") 
    print("  • Proper model configuration and weights")
    print("  • Regular updates and community support")

if __name__ == "__main__":
    test_speechbrain_ecapa()