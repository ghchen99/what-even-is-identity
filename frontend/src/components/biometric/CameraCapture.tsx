'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { CameraOptions } from '@/types/biometric';

interface CameraCaptureProps {
  onCapture: (imageFile: File) => void;
  onError: (error: string) => void;
  options?: Partial<CameraOptions>;
  autoCapture?: boolean;
  captureInterval?: number; // milliseconds
  className?: string;
}

export default function CameraCapture({
  onCapture,
  onError,
  options = {},
  autoCapture = false,
  captureInterval = 2000,
  className = ''
}: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string>('');
  const [captureCount, setCaptureCount] = useState(0);

  const defaultOptions: CameraOptions = {
    width: 640,
    height: 480,
    facingMode: 'user',
    ...options
  };

  const startCamera = useCallback(async () => {
    try {
      setError('');
      
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: defaultOptions.width },
          height: { ideal: defaultOptions.height },
          facingMode: defaultOptions.facingMode
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsStreaming(true);

        // Start auto-capture if enabled
        if (autoCapture) {
          startAutoCapture();
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access camera';
      setError(errorMessage);
      onError(errorMessage);
    }
  }, [defaultOptions, autoCapture, onError]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsStreaming(false);
    setCaptureCount(0);
  }, []);

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth || defaultOptions.width;
    canvas.height = video.videoHeight || defaultOptions.height;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob and create File
    canvas.toBlob((blob) => {
      if (blob) {
        const timestamp = new Date().getTime();
        const file = new File([blob], `face-capture-${timestamp}.jpg`, {
          type: 'image/jpeg',
          lastModified: timestamp
        });
        onCapture(file);
        setCaptureCount(prev => prev + 1);
      }
    }, 'image/jpeg', 0.8);
  }, [isStreaming, defaultOptions, onCapture]);

  const startAutoCapture = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(captureImage, captureInterval);
  }, [captureImage, captureInterval]);

  const stopAutoCapture = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <Card className={`w-full max-w-md ${className}`}>
      <CardHeader>
        <CardTitle className="text-center">Face Capture</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Video Preview */}
        <div className="relative">
          <video
            ref={videoRef}
            className="w-full h-auto rounded-lg border"
            autoPlay
            playsInline
            muted
            style={{ display: isStreaming ? 'block' : 'none' }}
          />
          {!isStreaming && (
            <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg border">
              <p className="text-gray-500">Camera not active</p>
            </div>
          )}
        </div>

        {/* Hidden canvas for image capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Controls */}
        <div className="flex flex-col space-y-2">
          {!isStreaming ? (
            <Button onClick={startCamera} className="w-full">
              Start Camera
            </Button>
          ) : (
            <>
              <div className="flex space-x-2">
                <Button onClick={captureImage} className="flex-1">
                  Capture Image
                </Button>
                <Button onClick={stopCamera} variant="outline" className="flex-1">
                  Stop Camera
                </Button>
              </div>
              
              {autoCapture && (
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>Auto-capture: {intervalRef.current ? 'ON' : 'OFF'}</span>
                  <span>Captured: {captureCount}</span>
                </div>
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}