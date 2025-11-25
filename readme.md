# 🎭 Real-Time Speech Emotion Detector

97.46% accuracy emotion detection from live audio streams.

## Features
- Real-time continuous emotion detection
- 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
- WebSocket-based streaming
- Silence detection

## Tech Stack
- Python 3.11
- FastAPI + WebSockets
- Transformers (r-f/wav2vec-english-speech-emotion-recognition)
- PyTorch

## Local Setup

Create virtual environment
python3.11 -m venv emotion_env
emotion_env\Scripts\activate # Windows

Install dependencies
pip install -r requirements.txt

Run
python app.py


## Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

## Usage
1. Open http://localhost:8000
2. Click START
3. Allow microphone access
4. Speak with emotion!

6. Push to GitHub
# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Real-time emotion detector with 97% accuracy"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/emotion-detector.git

# Push
git branch -M main
git push -u origin main

7. For Render Deployment
Create render.yaml:
services:
  - type: web
    name: emotion-detector
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
