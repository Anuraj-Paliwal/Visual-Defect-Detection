import cv2
import numpy as np

img = cv2.imread("defect4.png")

if img is None:
    print("Error: Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# STEP 1: Remove edges (logos, outlines)
# -----------------------------
edges = cv2.Canny(gray, 100, 200)
edges = cv2.dilate(edges, np.ones((5,5), np.uint8))  # expand edges

# -----------------------------
# STEP 2: Get smooth surface
# -----------------------------
blur = cv2.GaussianBlur(gray, (51, 51), 0)

# Dent map (low-frequency difference)
dent_map = blur - gray
dent_map = cv2.normalize(dent_map, None, 0, 255, cv2.NORM_MINMAX)
dent_map = dent_map.astype(np.uint8)

# -----------------------------
# STEP 3: Remove edge influence
# -----------------------------
dent_map[edges > 0] = 0   # kill edges completely

# -----------------------------
# STEP 4: Strong smoothing (IMPORTANT)
# -----------------------------
dent_smooth = cv2.GaussianBlur(dent_map, (21, 21), 0)

# -----------------------------
# STEP 5: Adaptive threshold (better than fixed)
# -----------------------------
mask = cv2.adaptiveThreshold(
    dent_smooth,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,
    -5
)

# -----------------------------
# STEP 6: Clean noise
# -----------------------------
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# STEP 7: Final heatmap (ONLY dents)
# -----------------------------
heatmap = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
heatmap[:] = [255, 0, 0]   # Blue background
heatmap[mask == 255] = [0, 0, 255]  # Red dents

# -----------------------------
# STEP 8: Show both
# -----------------------------
texture_display = cv2.cvtColor(dent_smooth, cv2.COLOR_GRAY2BGR)
combined = np.hstack((texture_display, heatmap))

cv2.imshow("Texture (Left) | Heatmap (Right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()