'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { VoiceRecorderOptions } from '@/types/biometric';

interface VoiceRecorderProps {
  onRecordingComplete: (audioFile: File) => void;
  onError: (error: string) => void;
  options?: Partial<VoiceRecorderOptions>;
  className?: string;
}

export default function VoiceRecorder({
  onRecordingComplete,
  onError,
  options = {},
  className = ''
}: VoiceRecorderProps) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string>('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioURL, setAudioURL] = useState<string>('');

  const defaultOptions: VoiceRecorderOptions = {
    sampleRate: 16000,
    bitRate: 128000,
    maxDuration: 10, // 10 seconds
    ...options
  };

  const startRecording = useCallback(async () => {
    try {
      setError('');
      setRecordingTime(0);
      setAudioURL('');
      chunksRef.current = [];

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: defaultOptions.sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      streamRef.current = stream;

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const timestamp = new Date().getTime();
        
        // Create audio URL for playback
        const url = URL.createObjectURL(blob);
        setAudioURL(url);

        // Create File object
        const file = new File([blob], `voice-recording-${timestamp}.webm`, {
          type: 'audio/webm',
          lastModified: timestamp
        });

        onRecordingComplete(file);
      };

      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);

      // Start timer
      let seconds = 0;
      timerRef.current = setInterval(() => {
        seconds += 1;
        setRecordingTime(seconds);
        
        // Auto-stop at max duration
        if (seconds >= defaultOptions.maxDuration) {
          stopRecording();
        }
      }, 1000);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access microphone';
      setError(errorMessage);
      onError(errorMessage);
    }
  }, [defaultOptions, onRecordingComplete, onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setIsRecording(false);
  }, [isRecording]);

  const clearRecording = useCallback(() => {
    if (audioURL) {
      URL.revokeObjectURL(audioURL);
      setAudioURL('');
    }
    setRecordingTime(0);
    chunksRef.current = [];
  }, [audioURL]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioURL) {
        URL.revokeObjectURL(audioURL);
      }
    };
  }, [audioURL]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const progressPercentage = (recordingTime / defaultOptions.maxDuration) * 100;

  return (
    <Card className={`w-full max-w-md ${className}`}>
      <CardHeader>
        <CardTitle className="text-center">Voice Recording</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}


        {/* Recording Status */}
        <div className="text-center space-y-2">
          <div className="text-2xl font-mono">
            {formatTime(recordingTime)}
          </div>
          <Progress value={progressPercentage} className="w-full" />
          <div className="text-sm text-gray-500">
            Max: {formatTime(defaultOptions.maxDuration)}
          </div>
        </div>

        {/* Recording Indicator */}
        {isRecording && (
          <div className="flex items-center justify-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <span className="text-red-500 font-medium">Recording...</span>
          </div>
        )}

        {/* Audio Playback */}
        {audioURL && (
          <div className="space-y-2">
            <audio controls className="w-full">
              <source src={audioURL} type="audio/webm" />
              Your browser does not support audio playback.
            </audio>
          </div>
        )}

        {/* Controls */}
        <div className="flex flex-col space-y-2">
          {!isRecording ? (
            <div className="flex space-x-2">
              <Button onClick={startRecording} className="flex-1">
                {recordingTime > 0 ? 'Record Again' : 'Start Recording'}
              </Button>
              {audioURL && (
                <Button onClick={clearRecording} variant="outline" className="flex-1">
                  Clear
                </Button>
              )}
            </div>
          ) : (
            <Button onClick={stopRecording} variant="destructive" className="w-full">
              Stop Recording
            </Button>
          )}
        </div>

        {/* Instructions */}
        <div className="text-sm text-gray-500 text-center space-y-1">
          {!isRecording && !audioURL && (
            <div>Click "Start Recording" to begin voice capture</div>
          )}
          {isRecording && (
            <div className="font-medium text-blue-600">
              Speak clearly into your microphone
            </div>
          )}
          {audioURL && (
            <div>Review your recording before submitting</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}