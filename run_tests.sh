#!/bin/bash

# Biometric API Test Runner
# This script starts the backend server and runs the API tests

echo "🚀 Biometric API Test Runner"
echo "============================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found"
    echo "Make sure you're running this from the project root"
    exit 1
fi

# Check for existing virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "📦 Using existing virtual environment at project root"
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "📦 Using existing virtual environment (venv) at project root"
    source venv/bin/activate
elif [ -f "backend/.venv/bin/activate" ]; then
    echo "📦 Using existing virtual environment in backend/"
    source backend/.venv/bin/activate
else
    echo "📦 Creating new virtual environment at project root..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✅ Virtual environment created"
fi

# Install/update dependencies
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Function to check if server is running
check_server() {
    curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1
    return $?
}

# Start backend server in background
echo "🖥️  Starting backend server..."
cd backend
python run.py &
SERVER_PID=$!
cd ..

# Wait for server to start
echo "⏳ Waiting for server to start..."
for i in {1..30}; do
    if check_server; then
        echo "✅ Server is running (PID: $SERVER_PID)"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "❌ Server failed to start within 30 seconds"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

# Run tests
echo ""
echo "🧪 Running API tests..."
cd "$(dirname "$0")"  # Go to script directory (project root)
python3 test_api.py

TEST_EXIT_CODE=$?

# Stop server
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "🎉 All tests completed successfully!"
    echo ""
    echo "📁 Check these directories for results:"
    echo "   - test-user/          (generated test files)"
    echo "   - backend/data/       (stored user data)"
else
    echo "❌ Some tests failed"
fi

exit $TEST_EXIT_CODE