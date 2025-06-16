'use client';

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import CameraCapture from './CameraCapture';
import VoiceRecorder from './VoiceRecorder';
import BiometricAPI from '@/lib/api';
import { BiometricEnrollmentResponse } from '@/types/biometric';
import { toast } from 'sonner';

interface BiometricEnrollmentProps {
  userId: string;
  onEnrollmentComplete?: (response: BiometricEnrollmentResponse) => void;
  className?: string;
}

export default function BiometricEnrollment({
  userId,
  onEnrollmentComplete,
  className = ''
}: BiometricEnrollmentProps) {
  const [faceImage, setFaceImage] = useState<File | null>(null);
  const [voiceAudio, setVoiceAudio] = useState<File | null>(null);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentResult, setEnrollmentResult] = useState<BiometricEnrollmentResponse | null>(null);
  const [error, setError] = useState<string>('');

  const handleFaceCapture = useCallback((imageFile: File) => {
    setFaceImage(imageFile);
    toast.success('Face image captured successfully');
  }, []);

  const handleVoiceCapture = useCallback((audioFile: File) => {
    setVoiceAudio(audioFile);
    toast.success('Voice recording completed');
  }, []);

  const handleError = useCallback((error: string) => {
    setError(error);
    toast.error(error);
  }, []);

  const performEnrollment = async () => {
    if (!faceImage || !voiceAudio) {
      setError('Please capture both face image and voice recording');
      return;
    }

    setIsEnrolling(true);
    setError('');

    try {
      const response = await BiometricAPI.enrollUser({
        user_id: userId,
        face_image: faceImage,
        voice_audio: voiceAudio
      });

      setEnrollmentResult(response);
      toast.success('Enrollment completed successfully!');
      
      if (onEnrollmentComplete) {
        onEnrollmentComplete(response);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Enrollment failed';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsEnrolling(false);
    }
  };

  const resetEnrollment = () => {
    setFaceImage(null);
    setVoiceAudio(null);
    setEnrollmentResult(null);
    setError('');
  };

  const enrollmentProgress = () => {
    let progress = 0;
    if (faceImage) progress += 40;
    if (voiceAudio) progress += 40;
    if (enrollmentResult) progress += 20;
    return progress;
  };

  if (enrollmentResult) {
    return (
      <Card className={`w-full max-w-2xl ${className}`}>
        <CardHeader>
          <CardTitle className="text-center text-green-600">
            ✅ Enrollment Successful
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-center space-y-2">
            <p className="text-lg font-medium">User enrolled successfully!</p>
            <div className="bg-green-50 p-4 rounded-lg space-y-2">
              <p><strong>User ID:</strong> {enrollmentResult.user_id}</p>
              <p><strong>Enrollment ID:</strong> {enrollmentResult.enrollment_id}</p>
              <p><strong>Status:</strong> {enrollmentResult.status}</p>
              <p><strong>Enrolled At:</strong> {new Date(enrollmentResult.enrolled_at).toLocaleString()}</p>
            </div>
          </div>
          <Button onClick={resetEnrollment} className="w-full">
            Enroll Another User
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`w-full max-w-4xl ${className}`}>
      <CardHeader>
        <CardTitle className="text-center">Biometric Enrollment</CardTitle>
        <div className="space-y-2">
          <div className="text-center text-sm text-gray-600">
            User ID: <span className="font-mono font-medium">{userId}</span>
          </div>
          <Progress value={enrollmentProgress()} className="w-full" />
          <div className="text-center text-xs text-gray-500">
            Progress: {enrollmentProgress()}%
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="face" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="face" className="flex items-center space-x-2">
              <span>📷</span>
              <span>Face Capture</span>
              {faceImage && <span className="text-green-500">✓</span>}
            </TabsTrigger>
            <TabsTrigger value="voice" className="flex items-center space-x-2">
              <span>🎤</span>
              <span>Voice Recording</span>
              {voiceAudio && <span className="text-green-500">✓</span>}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="face" className="space-y-4">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Capture Your Face</h3>
              <p className="text-sm text-gray-600 mb-4">
                Position your face clearly in the camera frame and click capture
              </p>
            </div>
            <div className="flex justify-center">
              <CameraCapture
                onCapture={handleFaceCapture}
                onError={handleError}
                options={{ width: 640, height: 480 }}
              />
            </div>
            {faceImage && (
              <div className="text-center">
                <Alert>
                  <AlertDescription>
                    ✅ Face image captured: {faceImage.name} ({Math.round(faceImage.size / 1024)}KB)
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </TabsContent>

          <TabsContent value="voice" className="space-y-4">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Record Your Voice</h3>
              <p className="text-sm text-gray-600 mb-4">
                Speak clearly for 3-5 seconds. Say something like "Hello, this is my voice"
              </p>
            </div>
            <div className="flex justify-center">
              <VoiceRecorder
                onRecordingComplete={handleVoiceCapture}
                onError={handleError}
                options={{ maxDuration: 10 }}
              />
            </div>
            {voiceAudio && (
              <div className="text-center">
                <Alert>
                  <AlertDescription>
                    ✅ Voice recorded: {voiceAudio.name} ({Math.round(voiceAudio.size / 1024)}KB)
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <Separator />

        {/* Enrollment Summary */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-center">Enrollment Summary</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className={`p-4 rounded-lg border ${faceImage ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
              <div className="text-center">
                <div className="text-2xl mb-2">📷</div>
                <div className="font-medium">Face Image</div>
                <div className="text-sm text-gray-600">
                  {faceImage ? 'Captured ✓' : 'Required'}
                </div>
              </div>
            </div>
            <div className={`p-4 rounded-lg border ${voiceAudio ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
              <div className="text-center">
                <div className="text-2xl mb-2">🎤</div>
                <div className="font-medium">Voice Recording</div>
                <div className="text-sm text-gray-600">
                  {voiceAudio ? 'Recorded ✓' : 'Required'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enrollment Button */}
        <div className="flex space-x-4">
          <Button
            onClick={performEnrollment}
            disabled={!faceImage || !voiceAudio || isEnrolling}
            className="flex-1"
          >
            {isEnrolling ? 'Processing Enrollment...' : 'Complete Enrollment'}
          </Button>
          <Button
            onClick={resetEnrollment}
            variant="outline"
            disabled={isEnrolling}
            className="flex-1"
          >
            Reset
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}