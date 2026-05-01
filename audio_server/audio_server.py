from flask import Flask, send_file, jsonify, make_response, render_template_string, request
import requests
import time
import os
import threading

app = Flask(__name__)

last_played_note = None
current_note = "unknown"

DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000", "unknown"]

def poll_currency_model():
    global current_note
    while True:
        try:
            response = requests.get("http://localhost:5001/api/status", timeout=2)
            if response.status_code == 200:
                data = response.json()
                current_note = data.get("result", "unknown")
        except Exception:
            current_note = "unknown"
        time.sleep(0.5)

@app.route('/', methods=['GET'])
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio Bridge — Currency Recognition</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 30px; }
            h1 { color: #bb86fc; text-align: center; margin-bottom: 6px; font-size: 26px; }
            .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }
            .status-box { background: #1e1e2e; border-radius: 12px; padding: 20px; margin-bottom: 30px;
                          border: 1px solid #333; text-align: center; }
            .status-box .label { color: #888; font-size: 13px; margin-bottom: 8px; }
            .status-box .val { font-size: 32px; font-weight: bold; color: #03dac6; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 16px; margin-bottom: 30px; }
            .btn { background: #1e1e2e; border: 2px solid #333; border-radius: 12px; padding: 20px 10px;
                   text-align: center; cursor: pointer; transition: all 0.2s; }
            .btn:hover { border-color: #bb86fc; background: #2a2a3e; transform: translateY(-3px); }
            .btn .amount { font-size: 28px; font-weight: bold; color: #03dac6; }
            .btn .lbl { font-size: 12px; color: #666; margin-top: 6px; }
            .btn.unknown-btn .amount { color: #ff6b6b; font-size: 18px; }
            .btn.unknown-btn { border-color: #ff6b6b55; }
            .btn.playing { border-color: #4caf50 !important; background: #1a2e1a; }
            .footer { text-align: center; color: #444; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>🔊 Audio Bridge Server</h1>
        <p class="subtitle">Currency Recognition — Click to test any denomination</p>

        <div class="status-box">
            <div class="label">Currently Detecting</div>
            <div class="val" id="live-detection">Loading...</div>
        </div>

        <div class="grid">
            {% for denom in denoms %}
            <div class="btn {% if denom == 'unknown' %}unknown-btn{% endif %}"
                 id="btn-{{ denom }}" onclick="playAudio('{{ denom }}')">
                <div class="amount">{% if denom != 'unknown' %}₹{{ denom }}{% else %}❓ Unknown{% endif %}</div>
                <div class="lbl">▶ Play</div>
            </div>
            {% endfor %}
        </div>

        <div class="footer">
            <p>ESP32 polls <code>/poll_audio</code> (WAV) &nbsp;|&nbsp; Browser uses <code>/audio/&lt;denom&gt;</code> (MP3)</p>
        </div>

        <script>
            let currentBtn = null;
            let currentAudio = null;

            function playAudio(denom) {
                if (currentAudio) { currentAudio.pause(); currentAudio = null; }
                if (currentBtn) currentBtn.classList.remove('playing');

                const btn = document.getElementById('btn-' + denom);
                btn.classList.add('playing');
                currentBtn = btn;

                const audio = new Audio('/audio/' + denom + '.mp3');
                currentAudio = audio;
                audio.play().catch(e => {
                    btn.classList.remove('playing');
                    alert('Playback error: ' + e.message);
                });
                audio.onended = () => btn.classList.remove('playing');
            }

            async function updateStatus() {
                try {
                    const r = await fetch('/status');
                    const d = await r.json();
                    const el = document.getElementById('live-detection');
                    if (d.current_detection !== 'unknown') {
                        el.textContent = '₹' + d.current_detection;
                        el.style.color = '#03dac6';
                    } else {
                        el.textContent = 'Unknown / No Note';
                        el.style.color = '#666';
                    }
                } catch(e) {}
            }
            setInterval(updateStatus, 1000);
            updateStatus();
        </script>
    </body>
    </html>
    """
    return render_template_string(html, denoms=DENOMINATIONS)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"current_detection": current_note, "last_played": last_played_note})

@app.route('/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    """Serve MP3 files for browser playback."""
    # Support both /audio/100 and /audio/100.mp3
    if not filename.endswith('.mp3'):
        filename = filename + '.mp3'
    file_path = f"audio_files/{filename}"
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/mpeg")
    return jsonify({"error": f"Audio file '{filename}' not found"}), 404

@app.route('/poll_audio', methods=['GET'])
def poll_audio():
    """ESP32 polls this. Returns WAV audio only when a new note is detected."""
    global last_played_note
    target_audio = current_note

    if target_audio != last_played_note:
        # WAV for ESP32 I2S
        wav_path = f"audio_files/{target_audio}.wav"
        if os.path.exists(wav_path):
            print(f"[Audio] Serving WAV: {target_audio}.wav")
            last_played_note = target_audio
            return send_file(wav_path, mimetype="audio/wav")

    return make_response("", 204)

if __name__ == '__main__':
    threading.Thread(target=poll_currency_model, daemon=True).start()
    print("Starting Audio Bridge Server on port 8000...")
    app.run(host='0.0.0.0', port=8000)
