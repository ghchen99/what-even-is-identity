# Biometric Identity System - Backend API

## 🏗️ Architecture Overview

FastAPI-based backend service implementing dual-modal biometric authentication using face and voice recognition. The system processes biometric data locally using ONNX Runtime and SpeechBrain models for secure, real-time identity verification.

## 🚀 Quick Setup

### Docker Deployment (Recommended)
```bash
# From project root
docker-compose up --build
```

### Manual Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download ML models
python download_models.py

# Run development server
python run.py
```

### Environment Variables
```bash
# Optional configuration
PRODUCTION=false          # Enable production mode
HOST=0.0.0.0             # Server host (default: 127.0.0.1)
PORT=8000                # Server port
CORS_ORIGINS=*           # Allowed CORS origins
```

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   └── config.py        # Application configuration
│   ├── api/
│   │   ├── routes.py        # API router configuration
│   │   └── endpoints/
│   │       ├── biometric.py # Biometric enrollment/verification
│   │       ├── health.py    # Health check endpoints
│   │       └── users.py     # User management (future)
│   ├── services/
│   │   ├── biometric.py     # Core biometric processing
│   │   ├── voice_speechbrain.py # Voice recognition service
│   │   └── storage.py       # Data persistence layer
│   ├── models/
│   │   └── user.py          # Data models and schemas
│   └── schemas/
│       └── biometric.py     # API request/response schemas
├── data/                    # Runtime data storage
│   ├── users/              # User profiles and embeddings
│   ├── verifications/      # Verification history
│   └── cache/              # Temporary data cache
├── models/                  # ML models directory
│   ├── facenet.onnx        # Face recognition model
│   └── spkrec-ecapa-voxceleb/ # Voice recognition models
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
├── run.py                  # Server entry point
└── Dockerfile              # Container configuration
```

## 🤖 Machine Learning Pipeline

### Face Recognition
**Model**: FaceNet (ONNX format)
- **Input**: 160x160 RGB images
- **Output**: 512-dimensional face embeddings
- **Preprocessing**:
  1. MediaPipe face detection
  2. Face cropping with padding
  3. Resize to 160x160 pixels
  4. Normalization (mean=0.5, std=0.5)
- **Fallback**: Center crop if face detection fails
- **Threshold**: 80% cosine similarity for verification

### Voice Recognition
**Model**: SpeechBrain ECAPA-TDNN
- **Input**: Audio files (MP3, WAV, WebM)
- **Output**: Speaker embeddings
- **Preprocessing**:
  1. Audio format conversion to WAV
  2. Resampling to 16kHz mono
  3. Feature extraction via SpeechBrain
- **Model Loading**: Automatic download from HuggingFace
- **Threshold**: 50% cosine similarity for verification

### Dual-Modal Verification
- **Security**: Both face AND voice must pass for authentication
- **Independent Processing**: Separate thresholds and scoring
- **Configurable**: Thresholds adjustable in config.py

## 🔧 Core Services

### BiometricService (`app/services/biometric.py`)
Primary service orchestrating all biometric operations:

**Key Methods**:
- `enroll_user()` - Process and store biometric templates
- `verify_user()` - Compare new samples against stored templates
- `extract_face_embedding()` - Face recognition pipeline
- `extract_voice_embedding()` - Voice recognition pipeline
- `calculate_similarity()` - Cosine similarity computation

**Features**:
- Lazy model loading for faster startup
- Debug mode with intermediate image saving
- Comprehensive error handling and logging
- Mock embeddings fallback for testing

### VoiceSpeechBrainService (`app/services/voice_speechbrain.py`)
Specialized voice processing service:

**Capabilities**:
- SpeechBrain model management
- Audio format conversion and preprocessing
- Speaker verification and embedding extraction
- Automatic model downloading and caching

### StorageService (`app/services/storage.py`)
File-based data persistence layer:

**Storage Types**:
- **User Profiles**: JSON files with biometric embeddings
- **Verification History**: TTL-based attempt logging
- **Cache**: Temporary data with automatic cleanup

**Features**:
- Thread-safe file operations
- Configurable data retention policies
- JSON serialization with custom encoders

## 🌐 API Endpoints

### Biometric Operations

#### POST `/api/v1/biometric/enroll`
Enroll a new user with biometric samples.

**Request**:
- `user_id` (form): Unique user identifier
- `face_image` (file): JPEG/PNG image (max 10MB)
- `voice_audio` (file): MP3/WAV/WebM audio (max 10MB)

**Response**:
```json
{
  "success": true,
  "enrollment_id": "uuid-string",
  "message": "User enrolled successfully",
  "user_id": "test-user-123"
}
```

#### POST `/api/v1/biometric/verify`
Verify user identity against stored biometric templates.

**Request**:
- `user_id` (form): User identifier to verify against
- `face_image` (file): JPEG/PNG image for verification
- `voice_audio` (file): Audio sample for verification

**Response**:
```json
{
  "success": true,
  "verified": true,
  "face_similarity": 0.85,
  "voice_similarity": 0.72,
  "face_threshold": 0.80,
  "voice_threshold": 0.50,
  "message": "Identity verified successfully"
}
```

### System Health

#### GET `/api/v1/health`
Check system status and model availability.

**Response**:
```json
{
  "status": "healthy",
  "face_model_loaded": true,
  "voice_model_loaded": true,
  "mediapipe_available": true,
  "storage_accessible": true
}
```

## ⚙️ Configuration

### Core Settings (`app/core/config.py`)
All configuration centralized using Pydantic Settings:

```python
# Model Configuration
FACE_MODEL_PATH = "models/facenet.onnx"
VOICE_MODEL_PATH = "models/ecapa_tdnn.onnx"

# Verification Thresholds
FACE_SIMILARITY_THRESHOLD = 0.80  # 80%
VOICE_SIMILARITY_THRESHOLD = 0.50  # 50%

# File Upload Limits
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10MB
ALLOWED_IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
ALLOWED_AUDIO_TYPES = {".mp3", ".wav", ".webm"}

# Storage Configuration
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
CACHE_TTL = 3600  # 1 hour
```

### Directory Structure Auto-Creation
The application automatically creates required directories:
- `data/users/` - User profiles
- `data/verifications/` - Verification history  
- `data/cache/` - Temporary cache
- `models/` - ML model storage

## 🔐 Security Features

### Data Protection
- **Local Processing**: All biometric data processed on-premise
- **No Cloud Transmission**: Models and embeddings stored locally
- **File Validation**: Strict upload limits and MIME type checking
- **Path Sanitization**: Secure file handling and storage

### Authentication Safeguards
- **Dual-Modal Requirement**: Both biometrics must pass
- **Configurable Thresholds**: Adjustable security levels
- **Attempt Logging**: Comprehensive audit trail
- **TTL Cleanup**: Automatic purging of old data

### Privacy Measures
- **Embedding Storage**: Raw biometric data not persisted
- **Anonymous Processing**: User IDs are client-controlled
- **Local Storage**: No external data transmission
- **Debug Controls**: Optional intermediate data saving

## 🧪 Testing

### Test Suite
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app

# Run specific test
python -m pytest tests/test_api.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Enrollment test (with test files)
curl -X POST http://localhost:8000/api/v1/biometric/enroll \
  -F "user_id=test-user" \
  -F "face_image=@tests/fixtures/images/user001-1.jpeg" \
  -F "voice_audio=@tests/fixtures/voice/user001-1.mp3"
```

## 📊 Performance Considerations

### Model Loading
- **Lazy Initialization**: Models loaded on first use
- **Memory Management**: Efficient ONNX Runtime usage
- **Caching**: Model instances cached after loading

### Processing Optimization
- **Batch Processing**: Multiple similarity comparisons
- **Efficient Preprocessing**: Optimized image/audio pipelines
- **Resource Management**: Proper cleanup and garbage collection

### Scalability Features
- **Stateless Design**: No server-side session storage
- **File-based Storage**: Easy to scale with distributed filesystems
- **Configurable Limits**: Adjustable based on hardware capacity

## 🚨 Troubleshooting

### Common Issues

**Models Not Loading**:
```bash
# Download models manually
python download_models.py
# Check models directory
ls -la models/
```

**Permission Errors**:
```bash
# Fix data directory permissions
chmod -R 755 data/
mkdir -p data/{users,verifications,cache}
```

**Memory Issues**:
- Reduce `MAX_FILE_SIZE` in config
- Ensure 4GB+ RAM available
- Monitor model loading logs

**CORS Issues**:
- Check `CORS_ORIGINS` environment variable
- Verify frontend URL matches allowed origins

### Debug Mode
Enable debug logging and intermediate file saving:
```python
# In app/core/config.py
DEBUG = True
SAVE_DEBUG_IMAGES = True
```

## 📈 Monitoring

### Health Monitoring
- System health endpoint at `/api/v1/health`
- Model availability checking
- Storage accessibility verification

### Logging
- Structured logging with timestamps
- Biometric processing pipeline logs
- Error tracking and debugging information
- Performance metrics and timing data

## 🔄 Future Enhancements

- **Database Integration**: PostgreSQL/MongoDB support
- **Authentication**: JWT-based API security
- **Batch Processing**: Multi-user enrollment/verification
- **Analytics**: Advanced metrics and reporting
- **Cloud Storage**: Azure/AWS integration options
- **Horizontal Scaling**: Load balancer support