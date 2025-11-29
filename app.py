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

# Lazy load model to reduce startup memory
emotion_classifier = None

def get_model():
    global emotion_classifier
    if emotion_classifier is None:
        print("Loading model...")
        emotion_classifier = pipeline(
            "audio-classification",
            model="r-f/wav2vec-english-speech-emotion-recognition",
            device=-1  # CPU
        )
        print("✅ Model loaded!")
    return emotion_classifier

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
        .timestamp-box { background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; margin: 20px auto; max-width: 500px; }
        .timestamp { font-size: 16px; margin: 5px; color: #ffd700; font-family: monospace; }
        .lag { font-size: 18px; margin: 10px; color: #ff6b6b; font-weight: bold; }
        .emotions-breakdown { background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; margin: 20px auto; max-width: 500px; }
        .emotions-breakdown h4 { margin: 0 0 15px 0; font-size: 20px; color: #ffd700; }
        .emotion-bar { margin: 8px 0; text-align: left; }
        .emotion-label { font-size: 14px; margin-bottom: 3px; display: flex; justify-content: space-between; }
        .emotion-progress { background: rgba(255,255,255,0.2); border-radius: 10px; height: 20px; overflow: hidden; }
        .emotion-fill { height: 100%; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); transition: width 0.3s; }
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
        <div class="timestamp-box">
            <div class="timestamp" id="inputTime">Input: --</div>
            <div class="timestamp" id="processedTime">Processed: --</div>
            <div class="lag" id="lagTime">Lag: -- ms</div>
        </div>
        <div class="emotions-breakdown">
            <h4>📊 All Emotions</h4>
            <div id="emotionsBreakdown"></div>
        </div>
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
                        const receivedTimestamp = Date.now();
                        
                        if (data.silence) {
                            document.getElementById('status').innerText = '⚠️ SILENCE - Speak!';
                            document.getElementById('audioLevel').innerText = 'Level: ' + data.level + '%';
                        } else {
                            document.getElementById('emotion').innerText = emojiMap[data.emotion] || '🎭';
                            document.getElementById('label').innerText = data.emotion.toUpperCase();
                            document.getElementById('confidence').innerText = (data.confidence * 100).toFixed(1) + '%';
                            document.getElementById('audioLevel').innerText = 'Level: ' + data.level + '%';
                            document.getElementById('status').innerText = '✅ ' + data.emotion;
                            
                            // Display timestamps and calculate lag
                            document.getElementById('processedTime').innerText = 'Processed: ' + new Date(receivedTimestamp).toLocaleTimeString() + '.' + (receivedTimestamp % 1000).toString().padStart(3, '0');
                            
                            if (data.inputTimestamp) {
                                const lag = receivedTimestamp - data.inputTimestamp;
                                document.getElementById('lagTime').innerText = 'Lag: ' + lag + ' ms';
                            }
                            
                            // Display all emotions breakdown
                            if (data.allEmotions) {
                                // Create a static order for emotions
                                const emotionOrder = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
                                const emotionMap = {};
                                data.allEmotions.forEach(em => {
                                    emotionMap[em.label] = em.score;
                                });
                                
                                let html = '';
                                emotionOrder.forEach(emotionName => {
                                    const score = emotionMap[emotionName] || 0;
                                    const isPrimary = emotionName === data.emotion;
                                    const emoji = emojiMap[emotionName] || '🎭';
                                    const percent = (score * 100).toFixed(1);
                                    html += `
                                        <div class="emotion-bar">
                                            <div class="emotion-label">
                                                <span>${emoji} ${emotionName.toUpperCase()}${isPrimary ? ' ⭐' : ''}</span>
                                                <span>${percent}%</span>
                                            </div>
                                            <div class="emotion-progress">
                                                <div class="emotion-fill" style="width: ${percent}%"></div>
                                            </div>
                                        </div>
                                    `;
                                });
                                document.getElementById('emotionsBreakdown').innerHTML = html;
                            }
                        }
                    };

                    audioContext = new AudioContext({sampleRate: 16000});
                    source = audioContext.createMediaStreamSource(stream);
                    processor = audioContext.createScriptProcessor(16384, 1, 1);
                    
                    processor.onaudioprocess = (e) => {
                        if (ws.readyState === WebSocket.OPEN) {
                            const inputTimestamp = Date.now();
                            ws.send(JSON.stringify({
                                audio: Array.from(e.inputBuffer.getChannelData(0)),
                                timestamp: inputTimestamp
                            }));
                            document.getElementById('inputTime').innerText = 'Input: ' + new Date(inputTimestamp).toLocaleTimeString() + '.' + (inputTimestamp % 1000).toString().padStart(3, '0');
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
            parsed_data = json.loads(data)
            audio_chunk = np.array(parsed_data['audio'], dtype=np.float32)
            input_timestamp = parsed_data.get('timestamp')
            buffer.extend(audio_chunk)
            
            if len(buffer) >= 48000:
                audio = np.array(buffer[-48000:])
                audio_level = np.abs(audio).mean()
                
                if audio_level > 0.01:
                    audio = audio / max(np.abs(audio).max(), 1e-5)
                    
                    # Use pipeline - CORRECT and SIMPLE
                    classifier = get_model()
                    result = classifier(audio, sampling_rate=16000)
                    
                    # Sort emotions by confidence (descending)
                    sorted_emotions = sorted(result, key=lambda x: x['score'], reverse=True)
                    
                    await websocket.send_json({
                        "emotion": sorted_emotions[0]['label'],
                        "confidence": sorted_emotions[0]['score'],
                        "allEmotions": sorted_emotions,
                        "silence": False,
                        "level": int(min(audio_level * 1000, 100)),
                        "inputTimestamp": input_timestamp
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
    import os
    
    # Use PORT from environment (for Render) or default to 8000 (for local)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

