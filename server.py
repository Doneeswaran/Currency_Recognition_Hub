from flask import Flask, request, render_template_string
import random
import time
import os
import threading
import cv2
import numpy as np

recognition_history = []

app = Flask(__name__)
UPLOAD_FOLDER = 'captures'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Advanced Recognition: SIFT Feature Matching (More robust than ORB)
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

# FLANN Matcher for SIFT (Faster and more accurate than Brute Force for large feature sets)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Calibration: Hue ranges for Indian Rupee Notes (HSV)
# Tuned for ESP32-CAM greenish tint and sensor noise
COLOR_RANGES = {
    "10":   {"h": [10, 25],   "s": [50, 255],  "v": [50, 255]}, # Brown
    "20":   {"h": [35, 85],   "s": [50, 255],  "v": [50, 255]}, # Greenish-Yellow
    "50":   {"h": [95, 115],  "s": [70, 255],  "v": [70, 255]}, # Cyan/Blue
    "100":  {"h": [100, 165], "s": [10, 255],  "v": [40, 255]}, # Lavender (Adjusted for low-light/camera noise)
    "200":  {"h": [15, 35],   "s": [100, 255], "v": [100, 255]},# Bright Orange
    "500":  {"h": [75, 105],  "s": [0, 45],    "v": [30, 180]}, # Stone Grey (Tightened to avoid desaturated lavender)
    "2000": {"h": [160, 175], "s": [50, 255],  "v": [50, 255]} # Magenta
}

# Pre-load reference patterns
REFERENCE_LIBRARY = {}
REF_DIR = "references"

def load_references():
    print("Initializing SIFT Pattern Recognition Library...")
    if not os.path.exists(REF_DIR):
        os.makedirs(REF_DIR)
        return
    
    # Priority sorting to pick .png over .jpg (fixes broken ref_100.jpg issue)
    files = sorted(os.listdir(REF_DIR), key=lambda x: (x.split('.')[0], 0 if x.endswith('.png') else 1))
    processed_denoms = set()

    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            denom = filename.split('_')[1].split('.')[0]
            if denom in processed_denoms and filename.endswith('.jpg'):
                continue # Skip .jpg if we already have a .png or better
            
            path = os.path.join(REF_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Check if image is valid (catches the 121-byte corrupted file issue)
            if img is not None and img.size > 1000:
                # Resolution Matching: Scale master to 800px width for SIFT detail
                h, w = img.shape
                new_w = 800
                new_h = int(h * (800 / w))
                img_scaled = cv2.resize(img, (new_w, new_h))
                
                # Apply CLAHE to reference to match real-time frame enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                img_ref_enhanced = clahe.apply(img_scaled)
                
                if denom not in REFERENCE_LIBRARY:
                    REFERENCE_LIBRARY[denom] = []
                
                # Synthetic Training: Create 3 views for each note (Original, -15 deg, +15 deg)
                angles = [0, -15, 15]
                for angle in angles:
                    if angle == 0:
                        view = img_ref_enhanced
                    else:
                        M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
                        view = cv2.warpAffine(img_ref_enhanced, M, (new_w, new_h))
                    
                    kp, des = sift.detectAndCompute(view, None)
                    if des is not None:
                        REFERENCE_LIBRARY[denom].append(des)
                        processed_denoms.add(denom)
                
                print(f"  - Trained ₹{denom} with SIFT (Source: {filename})")
            else:
                print(f"  [!] Skipping invalid/corrupted reference: {filename}")

load_references()

def white_balance(img):
    """Simple Gray World White Balance to fix ESP32-CAM greenish tint."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def process_currency(image_path):
    """
    Analyzes the image using a High-Precision Engine: 
    SIFT Patterns + Adaptive Note Localization + Weighted Decision Logic.
    """
    # Load image
    image_raw = cv2.imread(image_path)
    if image_raw is None:
        return "error", {}

    # --- Phase 1: Image Enhancement ---
    # Fix the constant 'greenish' cast from low-cost ESP32 sensors
    image_balanced = white_balance(image_raw)
    
    # --- Phase 2: Adaptive Note Localization ---
    gray = cv2.cvtColor(image_balanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    is_note_shape = False
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        total_area = image_raw.shape[0] * image_raw.shape[1]
        
        # --- Phase 2.1: Shape Analysis (Note is a Rectangle) ---
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h if h > 0 else 0
        if aspect_ratio < 1: aspect_ratio = 1.0/aspect_ratio
        extent = float(area)/(w*h) if (w*h) > 0 else 0
        
        # Currency notes have specific aspect ratio and high rectangularity
        # Adjusted: Relaxed extent (0.55) to allow for fingers/slants
        is_note_shape = (1.3 <= aspect_ratio <= 4.0) and (extent > 0.55)
        
        if area > (total_area * 0.08) and is_note_shape:
            pad = 20
            y1, y2 = max(0, y-pad), min(image_raw.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(image_raw.shape[1], x+w+pad)
            image_roi = image_balanced[y1:y2, x1:x2]
        else:
            # Center crop as fallback but keep is_note_shape=False
            h_raw, w_raw, _ = image_raw.shape
            cy, cx = h_raw // 2, w_raw // 2
            rh, rw = int(h_raw * 0.35), int(w_raw * 0.35)
            image_roi = image_balanced[max(0, cy-rh):min(h_raw, cy+rh), max(0, cx-rw):min(w_raw, cx+rw)]
    else:
        image_roi = image_balanced

    # --- Phase 3: Final Enhancement for Matching ---
    img_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    img_final = cv2.resize(img_enhanced, (800, 600))
    
    # --- Phase 4: SIFT Pattern Matching ---
    kp_test, des_test = sift.detectAndCompute(img_final, None)
    if des_test is None or len(des_test) < 10:
        return "unknown", {}
    
    pattern_scores = {}
    for denom, des_list in REFERENCE_LIBRARY.items():
        max_good = 0
        for des_ref in des_list:
            matches = flann.knnMatch(des_test, des_ref, k=2)
            # Relaxed Lowe's ratio to 0.75 for non-specimen capture noise
            good_matches = [m_pair[0] for m_pair in matches if len(m_pair)==2 and m_pair[0].distance < 0.75 * m_pair[1].distance]
            max_good = max(max_good, len(good_matches))
        pattern_scores[denom] = max_good

    # --- Phase 5: Color Verification (HSV) ---
    hsv_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    color_verification = {}
    for denom, cfg in COLOR_RANGES.items():
        lower = np.array([cfg["h"][0], cfg["s"][0], cfg["v"][0]])
        upper = np.array([cfg["h"][1], cfg["s"][1], cfg["v"][1]])
        mask = cv2.inRange(hsv_roi, lower, upper)
        color_verification[denom] = cv2.countNonZero(mask) / (hsv_roi.shape[0] * hsv_roi.shape[1] if (hsv_roi.shape[0] * hsv_roi.shape[1]) > 0 else 1)

    # --- Phase 6: Decision Fusion Logic ---
    final_decision = "unknown"
    sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
    best_denom, max_matches = sorted_patterns[0]
    runner_up_denom, runner_up_matches = sorted_patterns[1] if len(sorted_patterns) > 1 else ("none", 0)
    
    sorted_colors = sorted(color_verification.items(), key=lambda x: x[1], reverse=True)
    best_color_denom, max_color_density = sorted_colors[0]
    
    # Logic Correction: If SIFT thinks it's 500 but color density for 100 is higher, 
    # it's likely a desaturated 100 note (lavender looks grey).
    if best_denom == "500" and color_verification.get("100", 0) > max_color_density * 0.8:
        if color_verification.get("100", 0) > 0.05:
            best_denom = "100"
            print("  [Decision Logic] Re-routing 500 -> 100 based on color evidence")

    pattern_confidence = max_matches / (runner_up_matches + 1)
    color_agreement = (best_denom == best_color_denom)
    
    print(f"Decision Debug | SIFT: {best_denom}({max_matches}) | Color: {best_color_denom}({max_color_density:.2%}) | NoteShape: {is_note_shape}")
    
    # 1. High Certainty (Pattern match with any note)
    if max_matches >= 35:
        final_decision = best_denom
    
    # 2. Medium Certainty (Pattern + Color agreement)
    elif max_matches >= 15 and (color_agreement or max_matches > 50):
        final_decision = best_denom
        
    # 3. Shape-based Color Fallback (If pattern is weak but color and shape are strong)
    elif max_color_density > 0.12 and is_note_shape:
        final_decision = best_color_denom
        
    # 4. Emergency Fallback for desaturated 100 notes
    elif color_verification.get("100", 0) > 0.15:
        final_decision = "100"
            
    return final_decision, pattern_scores

latest_frame = None
latest_frame_time = 0
current_result = "unknown"
current_scores = {}
analysis_lock = threading.Lock()

def recognition_worker():
    global latest_frame, current_result, current_scores
    last_processed_time = 0
    
    while True:
        if latest_frame is not None and latest_frame_time > last_processed_time:
            # Copy frame to avoid race conditions
            with analysis_lock:
                frame_to_process = latest_frame
                process_start_time = latest_frame_time
            
            # Temporary file for SIFT
            temp_file = f"temp_analysis.jpg"
            with open(temp_file, 'wb') as f:
                f.write(frame_to_process)
            
            # Run SIFT
            result, scores = process_currency(temp_file)
            
            # Update global state
            current_result = result
            current_scores = scores
            last_processed_time = process_start_time
            
            # Record in history only if it's a real detection
            if result != "unknown":
                hist_file = f"{UPLOAD_FOLDER}/capture_{int(time.time())}.jpg"
                os.rename(temp_file, hist_file)
                recognition_history.insert(0, {
                    "image": hist_file.split('/')[-1],
                    "result": result,
                    "timestamp": time.time()
                })
                if len(recognition_history) > 50:
                    recognition_history.pop()
            
            print(f"Background Analysis: {result} ({scores.get(result, 0)} matches)")
            
        time.sleep(0.1) # Check for new frames 10 times a second

# Start background worker
threading.Thread(target=recognition_worker, daemon=True).start()

def gen_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.05) # ~20 FPS limit

@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize', methods=['POST'])
def recognize():
    global latest_frame, latest_frame_time, current_result
    # Update latest frame immediately for live streaming
    with analysis_lock:
        latest_frame = request.data
        latest_frame_time = time.time()
    
    # Return the last known result immediately (non-blocking)
    return {
        "result": current_result,
        "scores": current_scores
    }

@app.route('/api/status')
def get_status():
    return {
        "result": current_result,
        "scores": current_scores,
        "history_count": len(recognition_history)
    }

@app.route('/clear-history', methods=['POST'])
def clear_history():
    global recognition_history
    recognition_history = []
    # Also delete physical files in captures folder
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            if f.endswith(".jpg"):
                os.remove(os.path.join(UPLOAD_FOLDER, f))
    except Exception as e:
        print(f"Error clearing files: {e}")
    return {"status": "success"}

@app.route('/')
def dashboard():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Currency Recognition Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
            h1 { color: #bb86fc; text-align: center; }
            .history { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; margin-top: 30px; }
            .card { background: #1e1e1e; border-radius: 12px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); transition: transform 0.2s; }
            .card:hover { transform: translateY(-5px); }
            .card img { width: 100%; border-radius: 8px; margin-bottom: 10px; }
            .card .result { font-size: 24px; font-weight: bold; color: #03dac6; text-align: center; }
            .card .timestamp { font-size: 12px; color: #888; text-align: center; margin-top: 5px; }
            .no-data { text-align: center; margin-top: 100px; color: #888; }
            .refresh { text-align: center; margin-bottom: 20px; }
            button { background: #bb86fc; color: #121212; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; }
            button:hover { background: #9965f4; }
        </style>
        <script>
            // Real-time status polling
            async function updateStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    const statusEl = document.getElementById('live-result');
                    if (data.result !== 'unknown') {
                        statusEl.textContent = '₹' + data.result;
                        statusEl.style.color = '#03dac6';
                        statusEl.style.fontSize = '24px';
                    } else {
                        statusEl.textContent = 'Searching...';
                        statusEl.style.color = '#888';
                        statusEl.style.fontSize = '18px';
                    }
                    
                    // Show top matches details
                    const detailsEl = document.getElementById('live-details');
                    let detailsText = '';
                    const sorted = Object.entries(data.scores).sort((a,b) => b[1]-a[1]);
                    if (sorted.length > 0) {
                        detailsText = 'Top matches: ' + sorted.slice(0, 3).map(([k,v]) => `₹${k}(${v})`).join(', ');
                    }
                    detailsEl.textContent = detailsText;
                    
                } catch (e) {
                    console.error("Polling error:", e);
                }
            }
            setInterval(updateStatus, 500);
        </script>
    </head>
    <body>
        <h1>Currency Recognition Hub</h1>
        
        <!-- NEW: Live Monitor Section -->
        <div style="max-width: 800px; margin: 0 auto 30px; background: #1e1e1e; padding: 20px; border-radius: 12px; border: 2px solid #333;">
            <h2 style="margin-top: 0; color: #03dac6; font-size: 18px;">🔴 LIVE CAMERA MONITOR</h2>
            <div style="position: relative; width: 100%; height: 480px; background: #000; border-radius: 8px; overflow: hidden;">
                <img src="/video_feed" style="width: 100%; height: 100%; object-fit: contain;" onerror="this.style.display='none'; document.getElementById('no-signal').style.display='flex';">
                <div id="no-signal" style="display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; align-items: center; justify-content: center; color: #555; flex-direction: column;">
                    <span style="font-size: 48px;">📷</span>
                    <p>No signal from ESP32-CAM... Waiting for connection.</p>
                </div>
            </div>
            <div id="live-status" style="margin-top: 15px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #333; padding-top: 10px;">
                <div style="color: #888; font-size: 14px;">
                    Detection: <span id="live-result" style="font-weight: bold; color: #03dac6; font-size: 24px;">Waiting...</span>
                    <div id="live-details" style="font-size: 11px; color: #555; margin-top: 4px;"></div>
                </div>
                <div style="text-align: right; color: #555; font-size: 12px;">
                    Status: <span style="color: #03dac6;">Scanning</span><br>
                    Mode: Async SIFT
                </div>
            </div>
        </div>

        <div class="refresh">
            <button onclick="location.reload()">Refresh Feed</button>
            <button onclick="clearHistory()" style="background: #ff4444; color: white;">Clear History</button>
            <p>Auto-refreshing every 5 seconds...</p>
        </div>
        <script>
            async function clearHistory() {
                if(confirm("Confirm clear all history and images?")) {
                    await fetch('/clear-history', { method: 'POST' });
                    location.reload();
                }
            }
        </script>
        <div style="text-align: center; margin-bottom: 30px;">
             <a href="/camera" style="text-decoration: none;">
                <button style="background: #03dac6;">Open Webcam Tester</button>
             </a>
        </div>
        {% if history %}
            <div class="history">
                {% for item in history %}
                    <div class="card">
                        <img src="/captures/{{ item.image }}" alt="Captured Bill">
                        <div class="result">₹{{ item.result }}</div>
                        <div class="timestamp">{{ item.timestamp|datetime }}</div>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <p class="no-data">No currency recognition attempts yet. Waiting for ESP32-CAM images...</p>
        {% endif %}
    </body>
    </html>
    """
    import datetime
    def format_datetime(value):
        return datetime.datetime.fromtimestamp(value).strftime('%H:%M:%S')

    app.jinja_env.filters['datetime'] = format_datetime
    return render_template_string(html_template, history=recognition_history)

@app.route('/camera')
def camera_tester():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Webcam Currency Tester</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; text-align: center; }
            h1 { color: #bb86fc; }
            .container { max-width: 600px; margin: 0 auto; background: #1e1e1e; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
            video { width: 100%; border-radius: 8px; background: #000; margin-bottom: 20px; }
            .controls { margin-bottom: 20px; }
            button { background: #03dac6; color: #121212; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 16px; transition: background 0.3s; }
            button:hover { background: #018786; }
            .btn:disabled { background: #444; cursor: not-allowed; }
            .btn-live { background: #d32f2f; }
            .btn-live.active { background: #388e3c; box-shadow: 0 0 10px rgba(56, 142, 60, 0.5); }
            
            .scanning-indicator {
                display: none;
                align-items: center;
                justify-content: center;
                margin-top: 10px;
                color: #4CAF50;
                font-weight: bold;
                font-size: 14px;
            }
            .scanning-indicator .dot {
                height: 10px;
                width: 10px;
                background-color: #4CAF50;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.8); opacity: 0.5; }
                50% { transform: scale(1.2); opacity: 1; }
                100% { transform: scale(0.8); opacity: 0.5; }
            }
            #result-container { margin-top: 20px; padding: 15px; border-radius: 8px; display: none; }
            .result-text { font-size: 28px; font-weight: bold; color: #bb86fc; }
            .nav-link { margin-top: 20px; display: block; color: #03dac6; text-decoration: none; }
            .nav-link:hover { text-decoration: underline; }
            canvas { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Webcam Currency Tester</h1>
            <div id="setup-message" style="margin-bottom: 20px; padding: 15px; background: #333; border-radius: 8px;">
                <p>Click the button below to start your camera.</p>
            </div>
            <div id="debug-log" style="margin-bottom: 20px; padding: 10px; background: #000; color: #0f0; font-family: monospace; font-size: 12px; text-align: left; height: 100px; overflow-y: auto; border: 1px solid #333; border-radius: 5px;">
                > Ready. Click "Start Camera"...
            </div>
            <div style="margin-bottom: 20px;">
                <button id="start-btn" class="btn" onclick="initWebcam()">Start Camera</button>
                <button id="recognize-btn" class="btn" style="display: none;" onclick="captureAndRecognize()">Recognize Currency</button>
                <button id="live-toggle-btn" class="btn btn-live" style="display: none;" onclick="toggleLiveScan()">Enable Live Scan</button>
            </div>
            
            <div id="scanning-status" class="scanning-indicator">
                <span class="dot"></span> Scanning Currency...
            </div>
            <video id="webcam" autoplay playsinline style="display: none;"></video>
            <div id="result-container">
                <div>Detected Denomination:</div>
                <div class="result-text" id="result-val">---</div>
                <div id="score-details" style="margin-top: 15px; font-size: 14px; text-align: left; background: #2a2a2a; padding: 10px; border-radius: 5px; display: none;"></div>
                <div style="font-size: 11px; color: #888; margin-top: 10px;">Scoring based on unique pattern matches.</div>
            </div>
            <a href="/" class="nav-link">← Back to Dashboard</a>
        </div>
        <canvas id="canvas"></canvas>

        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const recognizeBtn = document.getElementById('recognize-btn');
            const startBtn = document.getElementById('start-btn');
            const setupMsg = document.getElementById('setup-message');
            const resultContainer = document.getElementById('result-container');
            const resultVal = document.getElementById('result-val');
            const scoreDetails = document.getElementById('score-details');
            const debugLog = document.getElementById('debug-log');

            let isScanning = false;
            let scanInterval = null;
            let isProcessing = false;

            function log(msg) {
                const entry = document.createElement('div');
                entry.textContent = `> ${msg}`;
                debugLog.appendChild(entry);
                debugLog.scrollTop = debugLog.scrollHeight;
                console.log(msg);
            }

            async function initWebcam() {
                log("Requesting camera access...");
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { facingMode: "environment", width: 640, height: 480 } 
                    });
                    video.srcObject = stream;
                    video.style.display = 'block';
                    startBtn.style.display = 'none';
                    recognizeBtn.style.display = 'inline-block';
                    document.getElementById('live-toggle-btn').style.display = 'inline-block';
                    setupMsg.style.display = 'none';
                    log("Camera started successfully.");
                } catch (err) {
                    log(`FAILURE: ${err.name} - ${err.message}`);
                    alert("Camera Error: " + err.message);
                }
            }

            function toggleLiveScan() {
                const btn = document.getElementById('live-toggle-btn');
                const status = document.getElementById('scanning-status');
                
                isScanning = !isScanning;
                
                if (isScanning) {
                    btn.textContent = "Disable Live Scan";
                    btn.classList.add('active');
                    status.style.display = 'flex';
                    recognizeBtn.disabled = true;
                    log("Live Scan enabled.");
                    scanInterval = setInterval(captureAndRecognize, 1500);
                } else {
                    btn.textContent = "Enable Live Scan";
                    btn.classList.remove('active');
                    status.style.display = 'none';
                    recognizeBtn.disabled = false;
                    log("Live Scan disabled.");
                    if (scanInterval) clearInterval(scanInterval);
                }
            }

            async function captureAndRecognize() {
                if (isProcessing) return;
                isProcessing = true;
                
                try {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    
                    canvas.toBlob(blob => {
                        log("Capturing frame...");
                        fetch('/recognize', {
                            method: 'POST',
                            body: blob,
                            headers: { 
                                'Content-Type': 'image/jpeg',
                                'X-Requested-With': 'XMLHttpRequest',
                                'Accept': 'application/json'
                            }
                        })
                        .then(res => res.json())
                        .then(data => {
                            resultContainer.style.display = 'block';
                            if(data.result === 'unknown') {
                                resultVal.innerText = 'Searching...';
                                resultVal.style.color = '#888';
                            } else {
                                resultVal.innerText = '₹' + data.result;
                                resultVal.style.color = '#03dac6';
                                log(`DETECTED: ₹${data.result}`);
                            }
                            
                            let scoresHtml = '<strong>Pattern Match Breakdown:</strong><br>';
                            const sortedScores = Object.entries(data.scores).sort((a,b) => b[1] - a[1]);
                            sortedScores.forEach(([denom, score]) => {
                                scoresHtml += `₹${denom}: ${score} matches<br>`;
                            });
                            scoreDetails.innerHTML = scoresHtml;
                            scoreDetails.style.display = 'block';
                            isProcessing = false;
                        })
                        .catch(err => {
                            log("Communication Error!");
                            resultVal.textContent = "Error";
                            isProcessing = false;
                        });
                    }, 'image/jpeg');
                } catch (err) {
                    log(`Process Error: ${err.message}`);
                    isProcessing = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/captures/<filename>')
def get_image(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print("Starting Currency Recognition Pro Server...")
    app.run(host='0.0.0.0', port=5001)
