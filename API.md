# Biometric Unlock API Documentation

## Overview

This FastAPI backend provides biometric authentication using face and voice recognition. It uses local folder storage to mock Redis and PostgreSQL databases for development purposes.

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── biometric.py    # Enrollment & verification endpoints
│   │   │   ├── users.py        # User management endpoints
│   │   │   └── health.py       # Health check endpoints
│   │   └── routes.py           # API router configuration
│   ├── core/
│   │   └── config.py           # Application configuration
│   ├── models/
│   │   └── user.py             # Data models
│   ├── schemas/
│   │   └── biometric.py        # Pydantic schemas
│   ├── services/
│   │   ├── biometric.py        # ONNX model inference
│   │   └── storage.py          # File-based storage
│   └── main.py                 # FastAPI app creation
├── data/                       # Local storage (mock databases)
├── models/                     # ONNX model files
├── requirements.txt
├── run.py                      # Server startup script
└── .env.example               # Environment configuration
```

## Quick Start

1. Navigate to backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy and configure environment:
```bash
cp .env.example .env
# Edit .env file as needed
```

4. Start the server:
```bash
python run.py
```

5. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

#### `POST /api/v1/biometric/enroll`
Enroll a new user with face and voice biometrics.

**Parameters:**
- `user_id` (form): Unique identifier for the user
- `face_image` (file): Face image (JPEG/PNG)
- `voice_audio` (file): Voice audio sample

**Response:**
```json
{
  "status": "success",
  "enrollment_id": "uuid-string",
  "message": "User user123 enrolled successfully",
  "user_id": "user123"
}
```

#### `POST /api/v1/biometric/verify`
Verify a user's identity using face and voice biometrics.

**Parameters:**
- `user_id` (form): User identifier
- `face_image` (file): Face image for verification
- `voice_audio` (file): Voice audio for verification
- `threshold` (form, optional): Similarity threshold (default: 0.8)

**Response:**
```json
{
  "status": "success",
  "verified": true,
  "face_similarity": 0.92,
  "voice_similarity": 0.87,
  "combined_score": 0.90,
  "threshold": 0.8,
  "verification_id": "uuid-string"
}
```

### User Management

#### `GET /api/v1/users/{user_id}/enrollments`
Get all enrollments for a user.

#### `GET /api/v1/users/{user_id}/verifications`
Get recent verification attempts for a user.

**Query Parameters:**
- `limit` (optional): Number of recent verifications to return (default: 10)

#### `DELETE /api/v1/users/{user_id}`
Delete all data for a user.

### System Endpoints

#### `GET /api/v1/health`
Health check endpoint with model status.

#### `GET /api/v1/health/models`
Get detailed information about loaded ONNX models.

#### `POST /api/v1/health/cache/cleanup`
Clean up expired cache entries.

## Data Storage Structure

The system uses local folders to mock cloud databases:

```
backend/data/
├── users/              # Mock PostgreSQL - User enrollments
│   ├── user123.json
│   └── user456.json
├── verifications/      # Mock PostgreSQL - Verification history
│   ├── user123_verifications.json
│   └── user456_verifications.json
└── cache/             # Mock Redis - Temporary cache
    ├── temp_key1.json
    └── temp_key2.json
```

## ONNX Models

Place your ONNX models in the `backend/models/` directory:

- `backend/models/facenet.onnx` - Face recognition model (FaceNet)
- `backend/models/xvector.onnx` - Voice recognition model (x-vector)

When models are not available, the system uses mock embeddings for testing.

## Model Requirements

### Face Model (FaceNet)
- Input: RGB image, 160x160 pixels
- Expected format: NCHW (batch, channels, height, width)
- Normalization: [-1, 1] range
- Output: 512-dimensional embedding

### Voice Model (x-vector)
- Input: Audio features (MFCC or raw audio)
- Expected sample rate: 16kHz
- Output: 256-dimensional embedding

## Similarity Scoring

The system uses cosine similarity to compare embeddings:

- **Face Weight**: 60% of final score
- **Voice Weight**: 40% of final score
- **Combined Score**: Weighted average of face and voice similarities
- **Verification**: Pass if combined score ≥ threshold

## Security Considerations

1. **Data Storage**: All biometric data is stored locally
2. **Embeddings Only**: Raw biometric data is processed immediately and not stored
3. **Configurable Thresholds**: Adjust security vs usability
4. **No Network Storage**: All data remains on local machine

## Testing Without Models

The system includes mock embeddings when ONNX models are not available:

- Mock face embeddings: 512-dimensional random vectors
- Mock voice embeddings: 256-dimensional random vectors
- Consistent similarity calculations for testing

## Example Usage

### Enroll a User
```bash
curl -X POST "http://localhost:8000/api/v1/biometric/enroll" \
  -F "user_id=john_doe" \
  -F "face_image=@face.jpg" \
  -F "voice_audio=@voice.wav"
```

### Verify a User
```bash
curl -X POST "http://localhost:8000/api/v1/biometric/verify" \
  -F "user_id=john_doe" \
  -F "face_image=@face_verify.jpg" \
  -F "voice_audio=@voice_verify.wav" \
  -F "threshold=0.85"
```

### Check Health
```bash
curl "http://localhost:8000/api/v1/health"
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `404`: User not found
- `422`: Validation error (invalid input)
- `500`: Internal server error

Error responses include detailed messages for debugging.