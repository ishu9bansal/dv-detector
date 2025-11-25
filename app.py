import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from transformers import pipeline
import numpy as np
import json

app = FastAPI()

# Use pipeline - it handles everything correctly
print("Loading model...")
emotion_classifier = pipeline(
    "audio-classification",
    model="r-f/wav2vec-english-speech-emotion-recognition",
    device=-1  # CPU
)

print("✅ Model loaded!")

# TEST THE MODEL IMMEDIATELY
print("\n🧪 Testing model with sample audio...")
test_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second of noise
try:
    test_result = emotion_classifier(test_audio, sampling_rate=16000)
    print(f"✅ Model works! Test prediction: {test_result[0]['label']} ({test_result[0]['score']*100:.1f}%)")
except Exception as e:
    print(f"❌ Model test FAILED: {e}")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Emotion Detector</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }
        .container { background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); max-width: 600px; margin: 0 auto; }
        #emotion { font-size: 150px; margin: 30px; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        #label { font-size: 60px; font-weight: bold; margin: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        #confidence { font-size: 35px; margin: 10px; }
        #audioLevel { font-size: 20px; margin: 10px; color: #00ff87; }
        button { padding: 20px 50px; font-size: 24px; cursor: pointer; border: none; border-radius: 50px; background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white; font-weight: bold; transition: transform 0.2s; margin: 20px; }
        button:hover { transform: scale(1.05); }
        #status { font-size: 20px; margin: 20px; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Real-Time Emotion Detector</h1>
        <h3>97.46% Accuracy - Pipeline Method</h3>
        <div id="emotion">🎭</div>
        <div id="label">READY</div>
        <div id="confidence">0%</div>
        <div id="audioLevel">Audio Level: 0%</div>
        <button onclick="toggleRecording()">START</button>
        <p id="status">Click START</p>
    </div>

    <script>
        let ws, audioContext, processor, source, isRecording = false;
        const emojiMap = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'};

        async function toggleRecording() {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                    ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onopen = () => { document.getElementById('status').innerText = '🎤 LISTENING!'; };
                    
                    ws.onmessage = (e) => {
                        const data = JSON.parse(e.data);
                        if (data.silence) {
                            document.getElementById('status').innerText = '⚠️ SILENCE - Speak!';
                            document.getElementById('audioLevel').innerText = 'Level: ' + data.level + '%';
                        } else {
                            document.getElementById('emotion').innerText = emojiMap[data.emotion] || '🎭';
                            document.getElementById('label').innerText = data.emotion.toUpperCase();
                            document.getElementById('confidence').innerText = (data.confidence * 100).toFixed(1) + '%';
                            document.getElementById('audioLevel').innerText = 'Level: ' + data.level + '%';
                            document.getElementById('status').innerText = '✅ ' + data.emotion;
                        }
                    };

                    audioContext = new AudioContext({sampleRate: 16000});
                    source = audioContext.createMediaStreamSource(stream);
                    processor = audioContext.createScriptProcessor(16384, 1, 1);
                    
                    processor.onaudioprocess = (e) => {
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({audio: Array.from(e.inputBuffer.getChannelData(0))}));
                        }
                    };
                    
                    source.connect(processor);
                    processor.connect(audioContext.destination);
                    
                    isRecording = true;
                    document.querySelector('button').innerText = 'STOP';
                    document.querySelector('button').style.background = 'linear-gradient(45deg, #ff6b6b 0%, #ee5a6f 100%)';
                } catch(err) { alert('Mic denied!'); }
            } else {
                if (processor) processor.disconnect();
                if (source) source.disconnect();
                if (audioContext) audioContext.close();
                if (ws) ws.close();
                isRecording = false;
                document.querySelector('button').innerText = 'START';
                document.querySelector('button').style.background = 'linear-gradient(45deg, #f093fb 0%, #f5576c 100%)';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = []
    
    try:
        while True:
            data = await websocket.receive_text()
            audio_chunk = np.array(json.loads(data)['audio'], dtype=np.float32)
            buffer.extend(audio_chunk)
            
            if len(buffer) >= 48000:
                audio = np.array(buffer[-48000:])
                audio_level = np.abs(audio).mean()
                
                if audio_level > 0.01:
                    audio = audio / max(np.abs(audio).max(), 1e-5)
                    
                    # Use pipeline - CORRECT and SIMPLE
                    result = emotion_classifier(audio, sampling_rate=16000)
                    
                    await websocket.send_json({
                        "emotion": result[0]['label'],
                        "confidence": result[0]['score'],
                        "silence": False,
                        "level": int(min(audio_level * 1000, 100))
                    })
                else:
                    await websocket.send_json({
                        "silence": True,
                        "level": int(min(audio_level * 1000, 100))
                    })
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
