# Biometric Unlock - Tech Stack & Architecture

## 🎯 Project Vision

Build a **biometric device unlock system** using face and voice recognition, focusing on **InferenceOps** rather than traditional MLOps. The system uses pre-trained models for inference and emphasizes fast similarity-based authentication.

## 🏗️ High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Mobile App    │    │  Inference API   │    │  Data Layer     │
│                 │    │                  │    │                 │
│ React Native    │◄──►│ FastAPI          │◄──►│ Azure Redis     │
│ ONNX Runtime    │    │ ONNX Runtime     │    │ PostgreSQL      │
│ Camera/Mic      │    │ Embedding Logic  │    │ Blob Storage    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 📱 MVP (Minimum Viable Product)

### **Goal**: Prove the concept works end-to-end

### Core Components:
- **Mobile App**: React Native with camera/microphone access
- **Inference API**: FastAPI service with ONNX Runtime
- **Pre-trained Models**: FaceNet (face) + x-vector (voice) in ONNX format
- **Data Storage**: PostgreSQL for embeddings, Redis for caching
- **Similarity Engine**: Cosine similarity with configurable thresholds

### MVP Tech Stack:
```yaml
Frontend:
  - React Native (Android focus)
  - ONNX Runtime Mobile
  - React Native Camera
  - Audio recording libraries

Backend:
  - FastAPI (Python)
  - ONNX Runtime (Python)
  - Redis client
  - PostgreSQL client

Infrastructure:
  - Azure Cache for Redis
  - Azure Database for PostgreSQL
  - Local ONNX model files

Models:
  - FaceNet (face embeddings)
  - x-vector (voice embeddings)
  - Stored as .onnx files
```

### MVP Capabilities:
- ✅ User enrollment (capture face + voice samples)
- ✅ Biometric verification (unlock attempt)
- ✅ Embedding storage and retrieval
- ✅ Configurable similarity thresholds
- ✅ Basic security and privacy measures

### MVP Limitations:
- ❌ No CI/CD pipeline
- ❌ No production monitoring
- ❌ Manual APK distribution
- ❌ No model versioning