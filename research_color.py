import cv2
import numpy as np
import os

# New Hybrid Config
CURRENCY_DATA = {
    "10":   {"hsv": [10, 50, 50],   "range": 15},  # Brown
    "20":   {"hsv": [60, 100, 100], "range": 25}, # Green-yellow
    "50":   {"hsv": [100, 150, 150],"range": 20}, # Blue
    "100":  {"hsv": [130, 80, 150], "range": 25}, # Lavender
    "200":  {"hsv": [30, 180, 180], "range": 15}, # Orange
    "500":  {"hsv": [90, 20, 120],  "range": 30}, # Grey-green
    "2000": {"hsv": [160, 150, 150],"range": 20}  # Magenta
}

def analyze_hybrid(path):
    img = cv2.imread(path)
    if img is None: return "error"
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    roi = hsv[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    
    color_scores = {}
    for denom, cfg in CURRENCY_DATA.items():
        base = np.array(cfg["hsv"])
        lower = np.clip(base - np.array([cfg["range"], 50, 50]), 0, 255)
        upper = np.clip(base + np.array([cfg["range"], 255, 255]), 0, 255)
        mask = cv2.inRange(roi, lower, upper)
        color_scores[denom] = cv2.countNonZero(mask)
        
    return color_scores

target = "captures/capture_5554.jpg" # This was a 100 note
print(f"Color scores for {target}: {analyze_hybrid(target)}")
