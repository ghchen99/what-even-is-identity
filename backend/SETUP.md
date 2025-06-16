# Biometric Unlock System Setup Guide

## 🚀 Quick Start

### 1. Download Models
```bash
# Download and convert ONNX models
python download_models.py
```

This script will:
- Download FaceNet model (or create a simple CNN fallback)
- Create x-vector voice recognition model
- Convert both to ONNX format
- Save to `backend/models/`

### 2. Start Backend
```bash
cd backend
python run.py
```

### 3. Prepare Test Data
Place your test files in this structure:
```
test-user/
├── images/     # Face images (JPEG/PNG)
└── voice/      # Voice recordings (MP3/WAV)
```

### 4. Test API
```bash
# Automated testing
./run_tests.sh

# Or manual testing  
python test_api.py
```

## 📁 Project Structure

```
├── backend/
│   ├── app/                    # FastAPI application
│   ├── models/                 # ONNX model files
│   │   ├── facenet.onnx       # Face recognition model
│   │   └── xvector.onnx       # Voice recognition model
│   ├── data/                  # Local storage (created at runtime)
│   └── requirements.txt       # Python dependencies
├── test-user/                 # Test data (created by test script)
├── download_models.py         # Model download script
├── test_api.py               # API testing script
└── run_tests.sh              # Automated test runner
```

## 🔧 Model Details

### FaceNet Model
- **Architecture**: InceptionResnetV1 (or simple CNN fallback)
- **Input**: RGB images, 160x160 pixels
- **Output**: 512-dimensional face embeddings
- **Format**: ONNX v11

### X-Vector Model  
- **Architecture**: Frame-level processing + statistics pooling
- **Input**: MFCC features, shape [batch, 80, time_steps]
- **Output**: 256-dimensional voice embeddings
- **Format**: ONNX v11

## 🔐 Security Model

### User Data Isolation
- Each user has separate enrollment data
- Verification only compares against that user's stored embeddings
- No cross-user data access

### Testing "Failed Verification"
The test script simulates **imposter attacks** by:
1. Enrolling user with legitimate biometrics
2. Attempting verification with different biometric samples
3. Expecting low similarity scores and verification failure

This tests the system's ability to reject unauthorized access attempts.

## 📊 Similarity Scoring

### Combined Score Calculation
```python
combined_score = (face_similarity * 0.6) + (voice_similarity * 0.4)
verified = combined_score >= threshold
```

### Typical Score Ranges
- **Legitimate user**: 0.7 - 0.95
- **Imposter attempt**: 0.1 - 0.5  
- **Default threshold**: 0.8

## 🛠️ Development Setup

### Virtual Environment (Best Practice)
```bash
# Create venv at project root (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies  
pip install -r backend/requirements.txt
```

### Alternative: Backend-only Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Environment Configuration
```bash
cp backend/.env.example backend/.env
# Edit .env file as needed
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Testing Workflow

1. **Model Download**: `python download_models.py`
2. **Start Server**: `cd backend && python run.py`
3. **Run Tests**: `python test_api.py`

### Test Sequence
1. Health check → Server status
2. User enrollment → Store biometric templates (uses first image + voice file)
3. Successful verification → Same user (uses second image + voice file)
4. Failed verification → Different person's biometrics (uses third image + voice file)
5. User management → Get enrollment/verification history

## 🔍 Troubleshooting

### Model Issues
```bash
# Re-download models
python download_models.py

# Check model files exist
ls -la backend/models/
```

### Server Issues
```bash
# Check server health
curl http://localhost:8000/api/v1/health

# View server logs
cd backend && python run.py
```

### Test Failures
```bash
# Check test data
ls -la test-user/

# Verify API endpoints
curl http://localhost:8000/docs
```

## 📈 Production Considerations

### Model Quality
- Use real pre-trained models for production
- Consider fine-tuning on your data
- Implement model versioning

### Security Enhancements
- Add authentication/authorization
- Implement rate limiting
- Use encrypted storage
- Add audit logging

### Performance Optimization
- Use GPU acceleration
- Implement model caching
- Add load balancing
- Monitor inference latency

## 🎯 Next Steps

1. **Mobile App**: React Native frontend
2. **Real Models**: Production-quality ONNX models
3. **Cloud Storage**: Replace local files with databases
4. **Authentication**: Add user auth and API keys
5. **Monitoring**: Add logging and metrics