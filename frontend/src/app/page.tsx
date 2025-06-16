'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import BiometricEnrollment from '@/components/biometric/BiometricEnrollment';
import BiometricVerification from '@/components/biometric/BiometricVerification';
import BiometricAPI from '@/lib/api';
import { BiometricEnrollmentResponse, BiometricVerificationResponse } from '@/types/biometric';
import { toast, Toaster } from 'sonner';

export default function Home() {
  const [userId, setUserId] = useState('test-user-123');
  const [threshold, setThreshold] = useState(0.8);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [healthInfo, setHealthInfo] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('enrollment');

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const health = await BiometricAPI.checkHealth();
      setIsHealthy(true);
      setHealthInfo(health);
      toast.success('Backend connected successfully');
    } catch (error) {
      setIsHealthy(false);
      setHealthInfo(null);
      toast.error('Failed to connect to backend. Please ensure the backend server is running.');
    }
  };

  const handleEnrollmentComplete = (response: BiometricEnrollmentResponse) => {
    toast.success(`Enrollment completed for user ${response.user_id}`);
    // Automatically switch to verification tab
    setTimeout(() => {
      setActiveTab('verification');
    }, 2000);
  };

  const handleVerificationComplete = (response: BiometricVerificationResponse) => {
    if (response.verified) {
      toast.success(`Identity verified! Score: ${(response.combined_score * 100).toFixed(1)}%`);
    } else {
      toast.error(`Verification failed. Score: ${(response.combined_score * 100).toFixed(1)}%`);
    }
  };

  const generateRandomUserId = () => {
    const timestamp = Date.now();
    const random = Math.floor(Math.random() * 1000);
    setUserId(`user-${timestamp}-${random}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="container mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-gray-900">
            Biometric Identity System
          </h1>
          <p className="text-lg text-gray-600">
            Secure user enrollment and verification using face and voice biometrics
          </p>
          
          {/* Backend Status */}
          <Card className="max-w-md mx-auto">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <span className="font-medium">Backend Status:</span>
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${isHealthy ? 'bg-green-500' : isHealthy === false ? 'bg-red-500' : 'bg-yellow-500'}`} />
                  <span className={`text-sm font-medium ${isHealthy ? 'text-green-600' : isHealthy === false ? 'text-red-600' : 'text-yellow-600'}`}>
                    {isHealthy ? 'Connected' : isHealthy === false ? 'Disconnected' : 'Checking...'}
                  </span>
                  <Button onClick={checkBackendHealth} variant="outline" size="sm">
                    Refresh
                  </Button>
                </div>
              </div>
              
              {healthInfo && (
                <div className="mt-4 text-xs text-gray-600 space-y-1">
                  <div>Face Model: {healthInfo.models?.face_model?.loaded ? '✅ Loaded' : '❌ Not Loaded'}</div>
                  <div>Voice Model: {healthInfo.models?.voice_model?.loaded ? '✅ Loaded' : '❌ Not Loaded'}</div>
                  <div>MediaPipe: {healthInfo.models?.face_detection?.mediapipe_available ? '✅ Available' : '❌ Not Available'}</div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Configuration */}
        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle className="text-center">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">User ID</label>
                <div className="flex space-x-2">
                  <Input
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    placeholder="Enter user ID"
                    className="flex-1"
                  />
                  <Button onClick={generateRandomUserId} variant="outline" size="sm">
                    Random
                  </Button>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Verification Threshold</label>
                <div className="flex items-center space-x-2">
                  <Input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-500">({(threshold * 100).toFixed(0)}%)</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Interface */}
        {isHealthy === false && (
          <Alert variant="destructive" className="max-w-2xl mx-auto">
            <AlertDescription>
              Cannot connect to backend server. Please ensure the backend is running on port 8000.
              <br />
              Run: <code className="bg-gray-100 px-2 py-1 rounded">cd backend && python run.py</code>
            </AlertDescription>
          </Alert>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-3">
            <TabsTrigger value="enrollment">📝 Enrollment</TabsTrigger>
            <TabsTrigger value="verification">🔍 Verification</TabsTrigger>
            <TabsTrigger value="continuous">🔄 Live Verify</TabsTrigger>
          </TabsList>

          <TabsContent value="enrollment" className="space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold">User Enrollment</h2>
              <p className="text-gray-600">
                Register a new user by capturing their face and voice biometrics
              </p>
            </div>
            
            {userId && isHealthy && (
              <div className="flex justify-center">
                <BiometricEnrollment
                  userId={userId}
                  onEnrollmentComplete={handleEnrollmentComplete}
                />
              </div>
            )}
          </TabsContent>

          <TabsContent value="verification" className="space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold">Identity Verification</h2>
              <p className="text-gray-600">
                Verify user identity with a single face image and voice sample
              </p>
            </div>
            
            {userId && isHealthy && (
              <div className="flex justify-center">
                <BiometricVerification
                  userId={userId}
                  threshold={threshold}
                  onVerificationComplete={handleVerificationComplete}
                  continuousVerification={false}
                />
              </div>
            )}
          </TabsContent>

          <TabsContent value="continuous" className="space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold">Continuous Verification</h2>
              <p className="text-gray-600">
                Real-time identity verification with periodic face snapshots
              </p>
              <Alert>
                <AlertDescription>
                  This mode captures face images every 3 seconds for continuous identity verification.
                  Record your voice once, then let the camera run for ongoing verification.
                </AlertDescription>
              </Alert>
            </div>
            
            {userId && isHealthy && (
              <div className="flex justify-center">
                <BiometricVerification
                  userId={userId}
                  threshold={threshold}
                  onVerificationComplete={handleVerificationComplete}
                  continuousVerification={true}
                  verificationInterval={3000}
                />
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <Separator />
        <div className="text-center text-sm text-gray-500 space-y-2">
          <p>
            Biometric Identity System - Demo Application
          </p>
          <p>
            Face recognition powered by FaceNet | Voice recognition powered by ECAPA-TDNN
          </p>
        </div>
      </div>

      {/* Toast notifications */}
      <Toaster 
        position="top-right" 
        richColors 
        closeButton 
        duration={4000}
      />
    </div>
  );
}
