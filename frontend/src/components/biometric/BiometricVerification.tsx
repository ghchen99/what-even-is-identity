'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import CameraCapture from './CameraCapture';
import VoiceRecorder from './VoiceRecorder';
import BiometricAPI from '@/lib/api';
import { BiometricVerificationResponse } from '@/types/biometric';
import { toast } from 'sonner';

interface BiometricVerificationProps {
  userId: string;
  threshold?: number;
  onVerificationComplete?: (response: BiometricVerificationResponse) => void;
  continuousVerification?: boolean;
  verificationInterval?: number; // milliseconds
  className?: string;
}

interface VerificationAttempt {
  id: string;
  timestamp: Date;
  verified: boolean;
  combined_score: number;
  face_similarity: number;
  voice_similarity: number;
  // Dual verification fields
  face_verified: boolean;
  voice_verified: boolean;
}

export default function BiometricVerification({
  userId,
  threshold = 0.8,
  onVerificationComplete,
  continuousVerification = false,
  verificationInterval = 3000,
  className = ''
}: BiometricVerificationProps) {
  const [currentFaceImage, setCurrentFaceImage] = useState<File | null>(null);
  const [currentVoiceAudio, setCurrentVoiceAudio] = useState<File | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [verificationHistory, setVerificationHistory] = useState<VerificationAttempt[]>([]);
  const [latestResult, setLatestResult] = useState<BiometricVerificationResponse | null>(null);
  const [error, setError] = useState<string>('');
  const [successCount, setSuccessCount] = useState(0);
  const [totalAttempts, setTotalAttempts] = useState(0);

  const handleFaceCapture = useCallback((imageFile: File) => {
    setCurrentFaceImage(imageFile);
    
    // If we have both face and voice, and continuous verification is enabled, verify automatically
    if (currentVoiceAudio && continuousVerification && isActive) {
      performVerification(imageFile, currentVoiceAudio);
    }
  }, [currentVoiceAudio, continuousVerification, isActive]);

  const handleVoiceCapture = useCallback((audioFile: File) => {
    setCurrentVoiceAudio(audioFile);
    
    // If we have both face and voice, and continuous verification is enabled, verify automatically
    if (currentFaceImage && continuousVerification && isActive) {
      performVerification(currentFaceImage, audioFile);
    }
  }, [currentFaceImage, continuousVerification, isActive]);

  const handleError = useCallback((error: string) => {
    setError(error);
    toast.error(error);
  }, []);

  const performVerification = async (faceImage?: File, voiceAudio?: File) => {
    const face = faceImage || currentFaceImage;
    const voice = voiceAudio || currentVoiceAudio;

    if (!face || !voice) {
      setError('Both face image and voice recording are required');
      return;
    }

    setIsVerifying(true);
    setError('');

    try {
      const response = await BiometricAPI.verifyUser({
        user_id: userId,
        face_image: face,
        voice_audio: voice,
        threshold
      });

      setLatestResult(response);
      setTotalAttempts(prev => prev + 1);

      // Add to history
      const attempt: VerificationAttempt = {
        id: response.verification_id,
        timestamp: new Date(),
        verified: response.verified,
        combined_score: response.combined_score,
        face_similarity: response.face_similarity,
        voice_similarity: response.voice_similarity,
        face_verified: response.face_verified,
        voice_verified: response.voice_verified
      };

      setVerificationHistory(prev => [attempt, ...prev.slice(0, 9)]); // Keep last 10

      if (response.verified) {
        setSuccessCount(prev => prev + 1);
        toast.success(`✅ Dual Verification Successful! Face: ${response.face_verified ? '✓' : '✗'} | Voice: ${response.voice_verified ? '✓' : '✗'}`);
      } else {
        const failedModes = [];
        if (!response.face_verified) failedModes.push(`Face: ${(response.face_similarity * 100).toFixed(1)}% < ${(response.face_threshold * 100).toFixed(0)}%`);
        if (!response.voice_verified) failedModes.push(`Voice: ${(response.voice_similarity * 100).toFixed(1)}% < ${(response.voice_threshold * 100).toFixed(0)}%`);
        toast.error(`❌ Verification Failed - ${failedModes.join(' | ')}`);
      }

      if (onVerificationComplete) {
        onVerificationComplete(response);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Verification failed';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsVerifying(false);
    }
  };

  const startContinuousVerification = () => {
    setIsActive(true);
    setVerificationHistory([]);
    setSuccessCount(0);
    setTotalAttempts(0);
    toast.info('Continuous verification started');
  };

  const stopContinuousVerification = () => {
    setIsActive(false);
    toast.info('Continuous verification stopped');
  };

  const resetVerification = () => {
    setCurrentFaceImage(null);
    setCurrentVoiceAudio(null);
    setLatestResult(null);
    setVerificationHistory([]);
    setError('');
    setSuccessCount(0);
    setTotalAttempts(0);
    setIsActive(false);
  };


  const successRate = totalAttempts > 0 ? (successCount / totalAttempts) * 100 : 0;

  return (
    <Card className={`w-full max-w-6xl ${className}`}>
      <CardHeader>
        <CardTitle className="text-center">Biometric Verification</CardTitle>
        <div className="space-y-2">
          <div className="text-center text-sm text-gray-600">
            User ID: <span className="font-mono font-medium">{userId}</span> | 
            Threshold: <span className="font-medium">{(threshold * 100).toFixed(0)}%</span>
          </div>
          {totalAttempts > 0 && (
            <div className="text-center text-sm">
              Success Rate: <span className="font-medium text-green-600">{successRate.toFixed(1)}%</span> 
              ({successCount}/{totalAttempts})
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Latest Result */}
        {latestResult && (
          <Alert variant={latestResult.verified ? "default" : "destructive"}>
            <AlertDescription>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span>
                    {latestResult.verified ? '✅ Dual Verification Successful' : '❌ Dual Verification Failed'}
                  </span>
                  <span className="text-xs">
                    Combined: {(latestResult.combined_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className={`flex items-center justify-between p-2 rounded ${latestResult.face_verified ? 'bg-green-100' : 'bg-red-100'}`}>
                    <span>📷 Face</span>
                    <span className="font-medium">
                      {latestResult.face_verified ? '✓' : '✗'} {(latestResult.face_similarity * 100).toFixed(1)}%
                      <span className="text-xs text-gray-600 ml-1">
                        (≥{(latestResult.face_threshold * 100).toFixed(0)}%)
                      </span>
                    </span>
                  </div>
                  <div className={`flex items-center justify-between p-2 rounded ${latestResult.voice_verified ? 'bg-green-100' : 'bg-red-100'}`}>
                    <span>🎤 Voice</span>
                    <span className="font-medium">
                      {latestResult.voice_verified ? '✓' : '✗'} {(latestResult.voice_similarity * 100).toFixed(1)}%
                      <span className="text-xs text-gray-600 ml-1">
                        (≥{(latestResult.voice_threshold * 100).toFixed(0)}%)
                      </span>
                    </span>
                  </div>
                </div>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Capture Status */}
        <div className="grid grid-cols-2 gap-4">
          <div className={`p-4 rounded-lg border ${currentFaceImage ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
            <div className="text-center">
              <div className="text-2xl mb-2">📷</div>
              <div className="font-medium">Face Image</div>
              <div className="text-sm text-gray-600">
                {currentFaceImage ? `Captured ✓ (${Math.round(currentFaceImage.size / 1024)}KB)` : 'Required'}
              </div>
            </div>
          </div>
          <div className={`p-4 rounded-lg border ${currentVoiceAudio ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
            <div className="text-center">
              <div className="text-2xl mb-2">🎤</div>
              <div className="font-medium">Voice Recording</div>
              <div className="text-sm text-gray-600">
                {currentVoiceAudio ? `Recorded ✓ (${Math.round(currentVoiceAudio.size / 1024)}KB)` : 'Required'}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Camera Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-center">Live Camera</h3>
            <CameraCapture
              onCapture={handleFaceCapture}
              onError={handleError}
              autoCapture={continuousVerification && isActive}
              captureInterval={verificationInterval}
              options={{ width: 480, height: 360 }}
            />
          </div>

          {/* Voice Section */}
          <div className="space-y-4">
            <div className="text-center">
              <h3 className="text-lg font-medium">Voice Sample</h3>
              <p className="text-sm text-gray-600 mb-2">
                Speak clearly to verify your identity
              </p>
            </div>
            <VoiceRecorder
              onRecordingComplete={handleVoiceCapture}
              onError={handleError}
              options={{ maxDuration: 10 }}
            />
          </div>

          {/* Results Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-center">Recent Results</h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {verificationHistory.map((attempt) => (
                <div
                  key={attempt.id}
                  className={`p-3 rounded-lg border ${
                    attempt.verified ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center space-x-2">
                      <Avatar className="w-6 h-6">
                        <AvatarFallback className={attempt.verified ? 'bg-green-500' : 'bg-red-500'}>
                          {attempt.verified ? '✓' : '✗'}
                        </AvatarFallback>
                      </Avatar>
                      <span className="font-medium text-sm">
                        {(attempt.combined_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {attempt.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-xs text-gray-600 grid grid-cols-2 gap-2">
                    <span className={attempt.face_verified ? 'text-green-600' : 'text-red-600'}>
                      📷 {attempt.face_verified ? '✓' : '✗'} {(attempt.face_similarity * 100).toFixed(1)}%
                    </span>
                    <span className={attempt.voice_verified ? 'text-green-600' : 'text-red-600'}>
                      🎤 {attempt.voice_verified ? '✓' : '✗'} {(attempt.voice_similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
              {verificationHistory.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  No verification attempts yet
                </div>
              )}
            </div>
          </div>
        </div>

        <Separator />

        {/* Controls */}
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {continuousVerification ? (
              <>
                {!isActive ? (
                  <Button
                    onClick={startContinuousVerification}
                    disabled={!currentVoiceAudio}
                    className="w-full"
                  >
                    Start Continuous Verification
                  </Button>
                ) : (
                  <Button
                    onClick={stopContinuousVerification}
                    variant="destructive"
                    className="w-full"
                  >
                    Stop Verification
                  </Button>
                )}
              </>
            ) : (
              <Button
                onClick={() => performVerification()}
                disabled={!currentFaceImage || !currentVoiceAudio || isVerifying}
                className="w-full"
              >
                {isVerifying ? 'Verifying...' : 'Verify Identity'}
              </Button>
            )}
            
            <Button
              onClick={resetVerification}
              variant="outline"
              disabled={isVerifying}
              className="w-full"
            >
              Reset
            </Button>

            <div className="flex items-center justify-center text-sm text-gray-600">
              {isActive && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span>Active</span>
                </div>
              )}
            </div>
          </div>

          {/* Status Information */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-lg font-bold">{totalAttempts}</div>
              <div className="text-sm text-gray-600">Total Attempts</div>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <div className="text-lg font-bold text-green-600">{successCount}</div>
              <div className="text-sm text-gray-600">Successful</div>
            </div>
            <div className="p-3 bg-red-50 rounded-lg">
              <div className="text-lg font-bold text-red-600">{totalAttempts - successCount}</div>
              <div className="text-sm text-gray-600">Failed</div>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <div className="text-lg font-bold text-blue-600">{successRate.toFixed(1)}%</div>
              <div className="text-sm text-gray-600">Success Rate</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}