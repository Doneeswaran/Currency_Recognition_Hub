import cv2
import numpy as np
import os

# Copy the ranges from server.py to test
CURRENCY_RANGES = {
    "10":   {"lower": [0, 50, 50],   "upper": [15, 255, 200]},
    "20":   {"lower": [40, 40, 40],  "upper": [80, 255, 255]},
    "50":   {"lower": [90, 50, 50],  "upper": [110, 255, 255]},
    "100":  {"lower": [110, 30, 30], "upper": [150, 255, 255]},
    "200":  {"lower": [15, 100, 100], "upper": [35, 255, 255]},
    "500":  {"lower": [0, 0, 50],    "upper": [180, 40, 200]},
    "2000": {"lower": [150, 50, 50], "upper": [179, 255, 255]}
}

def create_test_image(color_hsv):
    """Creates a dummy 100x100 image with a specific HSV color."""
    hsv_img = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv_img[:] = color_hsv
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    cv2.imwrite("test_bill.jpg", bgr_img)
    return "test_bill.jpg"

def test_on_color(name, hsv_val):
    path = create_test_image(hsv_val)
    
    # Logic from server.py (simplified)
    img = cv2.imread(path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_size = img.shape[0] * img.shape[1]
    
    found = "unknown"
    for denom, bounds in CURRENCY_RANGES.items():
        lower = np.array(bounds["lower"])
        upper = np.array(bounds["upper"])
        mask = cv2.inRange(img_hsv, lower, upper)
        match_count = cv2.countNonZero(mask)
        if match_count > (roi_size * 0.5): # If > 50% matches
            found = denom
            break
            
    print(f"Testing {name} (HSV {hsv_val}): Detected as ₹{found}")
    return found == name

# Test cases
success = True
success &= test_on_color("100", [130, 100, 100]) # Lavender center
success &= test_on_color("2000", [160, 100, 100]) # Magenta center
success &= test_on_color("200", [25, 200, 200])   # Bright Yellow center

if success:
    print("\n✅ All basic color tests PASSED!")
else:
    print("\n❌ Some tests FAILED. Check HSV ranges.")
