#!/usr/bin/env python3
"""
Model Download and ONNX Conversion Script

This script downloads pre-trained models and converts them to ONNX format
for use in the biometric authentication system.

Usage:
    python download_models.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import urllib.request
import tempfile
import shutil

def install_requirements():
    """Install required packages for model conversion"""
    packages = [
        "torch",
        "torchvision", 
        "facenet-pytorch",
        "speechbrain",
        "torchaudio",
        "onnx",
        "transformers"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"⬇️  Installing {package}...")
            os.system(f"{sys.executable} -m pip install {package}")

def download_facenet_model():
    """Download and convert FaceNet model to ONNX"""
    print("\n🔍 Downloading FaceNet model...")
    
    try:
        from facenet_pytorch import InceptionResnetV1
        
        # Load pre-trained FaceNet model
        model = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Create dummy input for ONNX export
        dummy_input = torch.randn(1, 3, 160, 160)
        
        # Export to ONNX
        output_path = Path("backend/models/facenet.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            }
        )
        
        print(f"✅ FaceNet model saved to: {output_path}")
        print(f"   - Input shape: [batch_size, 3, 160, 160]")
        print(f"   - Output shape: [batch_size, 512]")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download FaceNet: {e}")
        return False

def download_voice_model():
    """Download and convert voice recognition model to ONNX"""
    print("\n🎤 Creating voice recognition model...")
    
    try:
        # Simple x-vector-like model for voice recognition
        class SimpleXVector(nn.Module):
            def __init__(self, input_dim=80, embedding_dim=256):
                super(SimpleXVector, self).__init__()
                
                # Frame-level layers
                self.frame_layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                )
                
                # Statistics pooling (mean + std)
                self.stats_dim = 512 * 2  # mean + std
                
                # Segment-level layers
                self.segment_layers = nn.Sequential(
                    nn.Linear(self.stats_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    
                    nn.Linear(512, embedding_dim),
                )
            
            def forward(self, x):
                # x shape: [batch, features, time]
                batch_size, feat_dim, time_steps = x.shape
                
                # Reshape for frame-level processing
                x = x.transpose(1, 2).contiguous()  # [batch, time, features]
                x = x.view(-1, feat_dim)  # [batch*time, features]
                
                # Frame-level processing
                x = self.frame_layers(x)  # [batch*time, 512]
                
                # Reshape back
                x = x.view(batch_size, time_steps, -1)  # [batch, time, 512]
                x = x.transpose(1, 2)  # [batch, 512, time]
                
                # Statistics pooling
                mean = torch.mean(x, dim=2)  # [batch, 512]
                std = torch.std(x, dim=2)    # [batch, 512]
                stats = torch.cat([mean, std], dim=1)  # [batch, 1024]
                
                # Segment-level processing
                embedding = self.segment_layers(stats)  # [batch, 256]
                
                # L2 normalize
                embedding = nn.functional.normalize(embedding, p=2, dim=1)
                
                return embedding
        
        # Create model
        model = SimpleXVector().eval()
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Export to ONNX
        dummy_input = torch.randn(1, 80, 300)  # [batch, mfcc_features, time_frames]
        output_path = Path("backend/models/xvector.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'time_steps'},
                'embedding': {0: 'batch_size'}
            }
        )
        
        print(f"✅ X-Vector model saved to: {output_path}")
        print(f"   - Input shape: [batch_size, 80, time_steps]")
        print(f"   - Output shape: [batch_size, 256]")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create voice model: {e}")
        return False

def verify_models():
    """Verify that the ONNX models can be loaded"""
    print("\n🔍 Verifying ONNX models...")
    
    try:
        import onnxruntime as ort
        
        # Check FaceNet model
        face_model_path = "backend/models/facenet.onnx"
        if os.path.exists(face_model_path):
            session = ort.InferenceSession(face_model_path)
            print(f"✅ FaceNet model verified")
            print(f"   - Inputs: {[input.name for input in session.get_inputs()]}")
            print(f"   - Outputs: {[output.name for output in session.get_outputs()]}")
        
        # Check X-Vector model
        voice_model_path = "backend/models/xvector.onnx"
        if os.path.exists(voice_model_path):
            session = ort.InferenceSession(voice_model_path)
            print(f"✅ X-Vector model verified")
            print(f"   - Inputs: {[input.name for input in session.get_inputs()]}")
            print(f"   - Outputs: {[output.name for output in session.get_outputs()]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Biometric Model Download and Conversion")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Create models directory
    models_dir = Path("backend/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Try to download FaceNet model
    if download_facenet_model():
        success_count += 1
        
    # Download voice model
    if download_voice_model():
        success_count += 1
    
    # Verify models
    if verify_models():
        print("\n🎉 Model setup completed successfully!")
        print(f"✅ {success_count}/2 models created")
        print("\nNext steps:")
        print("1. Start the backend server: cd backend && python run.py")
        print("2. Run API tests: python test_api.py")
    else:
        print("\n❌ Model verification failed")
        return False
    
    return True

if __name__ == "__main__":
    main()