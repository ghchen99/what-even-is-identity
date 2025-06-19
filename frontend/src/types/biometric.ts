export interface BiometricEnrollmentRequest {
  user_id: string;
  face_image: File;
  voice_audio: File;
}

export interface BiometricEnrollmentResponse {
  user_id: string;
  enrollment_id: string;
  status: string;
  enrolled_at: string;
}

export interface BiometricVerificationRequest {
  user_id: string;
  face_image: File;
  voice_audio: File;
}

export interface BiometricVerificationResponse {
  user_id: string;
  verification_id: string;
  verified: boolean;
  face_similarity: number;
  voice_similarity: number;
  verified_at: string;
  // Dual verification fields
  face_verified: boolean;
  voice_verified: boolean;
  face_threshold: number;
  voice_threshold: number;
}

export interface ApiError {
  detail: string;
}

export interface CameraOptions {
  width: number;
  height: number;
  facingMode: 'user' | 'environment';
}

export interface VoiceRecorderOptions {
  sampleRate: number;
  bitRate: number;
  maxDuration: number;
}