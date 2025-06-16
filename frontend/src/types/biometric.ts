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
  threshold?: number;
}

export interface BiometricVerificationResponse {
  user_id: string;
  verification_id: string;
  verified: boolean;
  face_similarity: number;
  voice_similarity: number;
  combined_score: number;
  threshold: number;
  verified_at: string;
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