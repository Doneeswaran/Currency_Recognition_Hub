import cv2
import os
import numpy as np

# Initialize ORB
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load references
references = {}
ref_dir = "references"
for filename in os.listdir(ref_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        denom = filename.split('_')[1].split('.')[0]
        img = cv2.imread(os.path.join(ref_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            references[denom] = des
            print(f"Loaded reference for ₹{denom} - {len(kp)} features")

def match_currency(test_img_path):
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return "error", {}
    
    # Preprocess (resize for speed)
    img = cv2.resize(img, (640, 480))
    kp_test, des_test = orb.detectAndCompute(img, None)
    
    if des_test is None: return "unknown", {}
    
    scores = {}
    for denom, des_ref in references.items():
        matches = bf.match(des_test, des_ref)
        # Filter "good" matches by distance
        good_matches = [m for m in matches if m.distance < 45]
        scores[denom] = len(good_matches)
        
    best_match = max(scores, key=scores.get)
    # Threshold for confidence
    if scores[best_match] > 15:
        return best_match, scores
    else:
        return "unknown", scores

# Test on one of the references themselves
test_file = "references/ref_100.png"
result, scores = match_currency(test_file)
print(f"\nTest Result: Detected as ₹{result}")
print(f"Scores breakdown: {scores}")
