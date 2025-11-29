import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from .audio_processor import AudioProcessor

app = FastAPI()
processor = AudioProcessor(sample_rate=16000, window_seconds=3)


# Serve UI from static file
with open("static/index.html", "r", encoding="utf-8") as f:
    INDEX_HTML = f.read()


@app.get("/")
async def get_root():
    return HTMLResponse(INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # WebSocket only coordinates IO; processing is delegated

    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            audio_chunk = np.array(parsed_data['audio'], dtype=np.float32)
            input_timestamp = parsed_data.get('timestamp')

            processor.add_chunk(audio_chunk)
            result = processor.process(input_timestamp=input_timestamp)

            if not result.get("ready"):
                continue

            await websocket.send_json(result)
    except Exception as e:
        print(f"Error: {e}")
