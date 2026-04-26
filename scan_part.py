"""
Automated Defect Detection — Inference
========================================
Loads the trained MobileNetV2 classifier and runs a sliding window
over an input image to localise and classify surface defects.

Usage:
    python scan_part.py <image_path>
    python scan_part.py NEU-DET/validation/images/crazing_1.jpg
"""

import sys
import os
import pickle
import numpy as np
import cv2
import tensorflow as tf

# ── Configuration ─────────────────────────────────────────────────────
IMAGE_SIZE    = 128
MODEL_PATH    = 'defect_inspector.keras'
ENCODER_PATH  = 'label_encoder.pkl'
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to report a detection

# Defect class colours (BGR) for bounding box display
CLASS_COLORS = {
    'crazing'        : (255, 100,   0),
    'inclusion'      : (  0, 200, 255),
    'patches'        : (  0, 255, 100),
    'pitted_surface' : (200,   0, 255),
    'rolled-in_scale': (255, 200,   0),
    'scratches'      : (  0,  80, 255),
}

# ── Load model and encoder ─────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

CLASSES = list(le.classes_)


def preprocess_crop(crop):
    """Resize, convert to RGB float32 and add batch dimension."""
    resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = rgb.astype('float32') / 255.0
    return np.expand_dims(norm, axis=0)


def sliding_window(image, step=40, window_sizes=None):
    """
    Generate (x, y, window) tuples by sliding windows of varying
    sizes across the image.
    """
    if window_sizes is None:
        h, w = image.shape[:2]
        window_sizes = [
            (int(h * 0.4), int(w * 0.4)),
            (int(h * 0.6), int(w * 0.6)),
            (int(h * 0.8), int(w * 0.8)),
        ]

    h, w = image.shape[:2]
    for win_h, win_w in window_sizes:
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                yield x, y, image[y:y + win_h, x:x + win_w]


def non_max_suppression(detections, iou_threshold=0.4):
    """
    Remove overlapping bounding boxes, keeping only the highest
    confidence detection in each overlapping group.
    """
    if not detections:
        return []

    boxes      = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
    scores     = np.array([d[4] for d in detections])
    class_ids  = [d[5] for d in detections]

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_threshold]

    return [detections[i] for i in keep]


def scan_image(image_path):
    """Run defect detection on a single image file."""
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at '{image_path}'")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Error: Could not read image '{image_path}'")
        return

    print(f"\nScanning: {image_path}")
    print(f"Image size: {image.shape[1]}×{image.shape[0]}px")
    print("Running sliding window detection...")

    # Collect all detections above threshold
    detections = []
    for x, y, window in sliding_window(image):
        if window.shape[0] < 20 or window.shape[1] < 20:
            continue
        crop = preprocess_crop(window)
        probs = model.predict(crop, verbose=0)[0]
        confidence = np.max(probs)
        class_idx  = np.argmax(probs)

        if confidence >= CONFIDENCE_THRESHOLD:
            x2 = x + window.shape[1]
            y2 = y + window.shape[0]
            detections.append((x, y, x2, y2, confidence, class_idx))

    # Apply Non-Maximum Suppression to remove overlapping boxes
    detections = non_max_suppression(detections)

    # ── Draw results ──────────────────────────────────────────────────
    display = image.copy()

    if not detections:
        status_text  = "PASSED QC"
        status_color = (0, 200, 0)
        print("\n  Result: NO DEFECTS DETECTED — PASSED QC")
    else:
        status_text  = f"DEFECT DETECTED ({len(detections)} region(s))"
        status_color = (0, 0, 220)
        print(f"\n  Result: {len(detections)} DEFECT(S) DETECTED")
        print(f"  {'Region':<8} {'Class':<20} {'Confidence':>10}")
        print(f"  {'-'*40}")

        for i, (x1, y1, x2, y2, conf, cls_idx) in enumerate(detections):
            cls_name = CLASSES[cls_idx]
            color    = CLASS_COLORS.get(cls_name, (0, 0, 255))

            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label     = f"{cls_name} ({conf*100:.0f}%)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            print(f"  #{i+1:<7} {cls_name:<20} {conf*100:>9.1f}%")

    # Status bar at top
    cv2.rectangle(display, (0, 0), (display.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(display, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # Show and save
    output_path = 'scan_result.jpg'
    cv2.imwrite(output_path, display)
    print(f"\n  Result image saved to '{output_path}'")

    cv2.imshow('NEU-DET Defect Scanner', display)
    print("  Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_part.py <image_path>")
        print("Example: python scan_part.py NEU-DET/validation/images/crazing_1.jpg")
        sys.exit(1)
    scan_image(sys.argv[1])
