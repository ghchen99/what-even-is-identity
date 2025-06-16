# Biometric Identity System - Frontend

A Next.js web application for testing the biometric identity system with face and voice recognition.

## Features

- **User Enrollment**: Capture face images and voice recordings to enroll new users
- **Identity Verification**: Verify user identity with face and voice biometrics
- **Continuous Verification**: Real-time identity verification with periodic face snapshots
- **Live Camera Feed**: Browser-based camera access for face capture
- **Voice Recording**: Browser-based microphone access for voice samples
- **Backend Integration**: RESTful API integration with the Python backend
- **Real-time Feedback**: Toast notifications and visual feedback for all operations

## Technology Stack

- **Framework**: Next.js 15 with TypeScript
- **UI Components**: shadcn/ui with Tailwind CSS
- **State Management**: React hooks
- **HTTP Client**: Axios
- **Notifications**: Sonner
- **File Handling**: React Dropzone
- **Media Capture**: Browser MediaDevices API

## Prerequisites

- Node.js 18+ and npm
- Modern web browser with camera and microphone access
- Backend server running (see ../backend/README.md)

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure backend URL** (optional):
   Create a `.env.local` file:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## Usage

### 1. Enrollment Mode
- Enter a user ID or generate a random one
- Switch to the "Enrollment" tab
- Capture a face image using the camera
- Record a voice sample (3-5 seconds)
- Click "Complete Enrollment" to register the user

### 2. Verification Mode
- Enter the user ID of an enrolled user
- Switch to the "Verification" tab
- Capture a face image and record voice
- Click "Verify Identity" to check authentication
- View similarity scores and verification result

### 3. Continuous Verification Mode
- Switch to the "Live Verify" tab
- Record a voice sample once
- Click "Start Continuous Verification"
- The camera will automatically capture images every 3 seconds
- View real-time verification results and success rate

## Features in Detail

### Camera Capture
- Live video preview
- Automatic face detection with MediaPipe (when available)
- Manual capture or automatic periodic capture
- Image preprocessing and cropping

### Voice Recording
- Browser-based audio recording
- Configurable recording duration
- Audio playback for review
- WebM audio format support

### Backend Integration
- Automatic backend health checking
- RESTful API communication
- File upload handling
- Error handling and user feedback

### Real-time Updates
- Toast notifications for all operations
- Progress indicators
- Live verification statistics
- Success/failure visual feedback

## Browser Permissions

The application requires:
- **Camera access**: For face image capture
- **Microphone access**: For voice recording

Grant permissions when prompted by your browser.

## Troubleshooting

### Backend Connection Issues
- Ensure the backend server is running on port 8000
- Check that CORS is properly configured
- Verify the API URL in environment variables

### Camera/Microphone Issues
- Grant browser permissions for media access
- Use HTTPS in production for media access
- Check browser compatibility for MediaDevices API

### Performance Issues
- Reduce verification interval in continuous mode
- Ensure adequate lighting for face detection
- Use a quiet environment for voice recording

## Browser Compatibility

- Chrome 80+ (recommended)
- Firefox 75+
- Safari 14+
- Edge 80+

## Development

### Project Structure
```
src/
├── app/                    # Next.js app router pages
├── components/
│   ├── ui/                # shadcn/ui components
│   └── biometric/         # Custom biometric components
├── lib/                   # Utility functions and API client
└── types/                 # TypeScript type definitions
```

### Adding New Components
```bash
npx shadcn@latest add [component-name]
```

### Building for Production
```bash
npm run build
npm start
```

## API Integration

The frontend communicates with the backend using these endpoints:
- `POST /api/v1/biometric/enroll` - User enrollment
- `POST /api/v1/biometric/verify` - Identity verification
- `GET /api/v1/health` - Backend health check

See the backend API documentation for detailed endpoint specifications.
