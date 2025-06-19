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
from pathlib import Path
import json

def install_requirements():
    """Install required packages for model conversion"""
    packages = [
        "torch",
        "torchvision", 
        "facenet-pytorch",
        "speechbrain",
        "torchaudio",
        "onnx",
        "transformers",
        "onnxruntime"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        try:
            if package == "speechbrain":
                # Special handling for speechbrain
                try:
                    import speechbrain
                    print(f"✅ {package} already installed")
                except ImportError:
                    print(f"⬇️  Installing {package}...")
                    os.system(f"{sys.executable} -m pip install git+https://github.com/speechbrain/speechbrain.git@develop")
            else:
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

def download_speechbrain_speaker_model():
    """Download SpeechBrain ECAPA-TDNN speaker model to backend/models"""
    print("\n🎤 Downloading SpeechBrain ECAPA-TDNN speaker model...")
    
    try:
        from speechbrain.inference.speaker import EncoderClassifier, SpeakerRecognition
        
        # Load pre-trained ECAPA-TDNN models from SpeechBrain
        print("   - Loading SpeechBrain ECAPA-TDNN encoder...")
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="backend/models/spkrec-ecapa-voxceleb"
        )
        
        print("   - Loading SpeechBrain speaker verification model...")
        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="backend/models/spkrec-ecapa-voxceleb"
        )
        
        print(f"✅ SpeechBrain models saved to: backend/models/spkrec-ecapa-voxceleb")
        print(f"   - Encoder for embedding extraction")
        print(f"   - Verification model for speaker comparison")
        print(f"   - Model trained on VoxCeleb dataset")
        print(f"   - EER: 0.69% on VoxCeleb1-test")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download SpeechBrain model: {e}")
        return False

def verify_models():
    """Verify that the models can be loaded"""
    print("\n🔍 Verifying models...")
    
    try:
        import onnxruntime as ort
        
        # Check FaceNet model
        face_model_path = "backend/models/facenet.onnx"
        if os.path.exists(face_model_path):
            session = ort.InferenceSession(face_model_path)
            print(f"✅ FaceNet model verified")
            print(f"   - Inputs: {[input.name for input in session.get_inputs()]}")
            print(f"   - Outputs: {[output.name for output in session.get_outputs()]}")
        else:
            print(f"❌ FaceNet model not found")
            return False
        
        # Check SpeechBrain model directory
        speechbrain_model_path = "backend/models/spkrec-ecapa-voxceleb"
        if os.path.exists(speechbrain_model_path):
            print(f"✅ SpeechBrain model directory verified")
            print(f"   - Path: {speechbrain_model_path}")
            print(f"   - Contains SpeechBrain ECAPA-TDNN models")
        else:
            print(f"❌ SpeechBrain model directory not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def create_model_info_file():
    """Create a JSON file with model information for the backend"""
    print("\n📝 Creating model information file...")
    
    model_info = {
        "face_model": {
            "name": "facenet",
            "path": "backend/models/facenet.onnx",
            "input_shape": [1, 3, 160, 160],
            "output_shape": [1, 512],
            "input_type": "image",
            "preprocessing": "resize_to_160x160_normalize",
            "description": "FaceNet model pretrained on VGGFace2 dataset"
        },
        "speaker_model": {
            "name": "speechbrain_ecapa_tdnn",
            "path": "backend/models/spkrec-ecapa-voxceleb",
            "model_type": "speechbrain",
            "input_type": "audio_waveform",
            "preprocessing": "speechbrain_internal",
            "description": "SpeechBrain ECAPA-TDNN model pretrained on VoxCeleb dataset",
            "performance": "0.69% EER on VoxCeleb1-test"
        }
    }
    
    # Save model info
    info_path = Path("backend/models/model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Model info saved to: {info_path}")

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
    
    # Download FaceNet model
    if download_facenet_model():
        success_count += 1
        
    # Download SpeechBrain ECAPA-TDNN model
    if download_speechbrain_speaker_model():
        success_count += 1
    
    # Verify models
    if verify_models() and success_count == 2:
        # Create model info file
        create_model_info_file()
        
        print("\n🎉 Model setup completed successfully!")
        print(f"✅ {success_count}/2 models created")
        print("\nDownloaded models:")
        print("- Face recognition: FaceNet (VGGFace2 pretrained)")
        print("- Speaker recognition: SpeechBrain ECAPA-TDNN (VoxCeleb pretrained)")
        print("- Performance: 0.69% EER on VoxCeleb1-test")
            
        print("\nNext steps:")
        print("1. Start the backend server: cd backend && python run.py")
        print("2. Run API tests: python test_api.py")
        print("3. Check model_info.json for integration details")
        
        return True
    else:
        print("\n❌ Model setup failed")
        print(f"Successfully created: {success_count}/2 models")
        return False

if __name__ == "__main__":
    main()