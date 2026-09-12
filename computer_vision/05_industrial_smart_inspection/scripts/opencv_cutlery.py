from pathlib import Path

import cv2
import numpy as np


CUTLERY_CN = {
    "kuaizi": "筷子",
    "shaozi": "勺子",
    "yaqian": "牙签",
    "zhijin": "纸巾",
}


def normalize_label(name):
    return str(name).split("（", 1)[0]


def read_image(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def clear_border(mask, margin_ratio=0.04):
    h, w = mask.shape[:2]
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    out = mask.copy()
    out[:my, :] = 0
    out[h - my :, :] = 0
    out[:, :mx] = 0
    out[:, w - mx :] = 0
    return out


def largest_box(mask, min_area=800, min_ratio=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = max(w, h) / max(1, min(w, h))
        if min_ratio is not None and ratio < min_ratio:
            continue
        candidates.append((area, (x, y, x + w, y + h)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def merge_boxes(boxes):
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


def contour_boxes(mask, min_area=800, min_ratio=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = max(w, h) / max(1, min(w, h))
        if min_ratio is not None and ratio < min_ratio:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def detect_spoon(hsv, gray):
    # Black spoon: low brightness, connected long handle + bowl.
    mask = cv2.inRange(gray, 0, 72)
    mask = clear_border(mask, 0.06)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return largest_box(mask, min_area=2500, min_ratio=1.6)


def detect_chopsticks(hsv, gray):
    # Brown chopsticks: warm hue, moderate saturation, long and thin shape.
    lower = np.array([5, 35, 35], dtype=np.uint8)
    upper = np.array([32, 190, 210], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = clear_border(mask, 0.03)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    boxes = contour_boxes(mask, min_area=900, min_ratio=4.0)
    if len(boxes) >= 1:
        return merge_boxes(boxes)
    return None


def detect_tissue(hsv, gray):
    # Tissue: bright, low-saturation, compact rectangular region.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat < 55) & (val > 95) & (gray > 88)).astype(np.uint8) * 255
    mask = clear_border(mask, 0.06)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)
        if ratio < 2.2 and fill > 0.35:
            candidates.append((area, (x, y, x + w, y + h)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def detect_toothpick(gray, existing_boxes):
    # Toothpick: slim straight line, often semi-transparent. Hough lines work
    # better than color thresholding under uneven lighting.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 10, 55)
    edges = clear_border(edges, 0.03)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=28, minLineLength=170, maxLineGap=70)
    if lines is None:
        return None

    good = []
    h, w = gray.shape[:2]
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min(w, h) * 0.2:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not (25 <= angle <= 65 or 75 <= angle <= 105):
            continue
        box = (min(x1, x2) - 18, min(y1, y2) - 18, max(x1, x2) + 18, max(y1, y2) + 18)
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        if max(width, height) / min(width, height) < 2.4:
            continue
        roi = gray[max(0, box[1]) : min(h, box[3]), max(0, box[0]) : min(w, box[2])]
        if roi.size == 0 or not (35 < float(roi.mean()) < 135):
            continue
        if overlaps_existing(box, existing_boxes, threshold=0.45):
            continue
        good.append((length, box))
    if not good:
        return None
    best = max(good, key=lambda item: item[0])[1]
    return best


def overlaps_existing(box, existing_boxes, threshold=0.3):
    for other in existing_boxes:
        x1 = max(box[0], other[0])
        y1 = max(box[1], other[1])
        x2 = min(box[2], other[2])
        y2 = min(box[3], other[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        if inter / area > threshold:
            return True
    return False


def yolo_present(rows, min_score=0.25):
    present = set()
    boxes = []
    for row in rows:
        if float(row.get("score", 0)) < min_score:
            continue
        base = normalize_label(row.get("name", ""))
        if base in CUTLERY_CN:
            present.add(base)
        if row.get("box"):
            boxes.append(tuple(int(float(v)) for v in row["box"]))
    return present, boxes


def compensate_cutlery(image_path, rows, plotted):
    img = read_image(image_path)
    if img is None:
        return plotted, []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    present, existing_boxes = yolo_present(rows, min_score=0.25)
    if len(present) < 2:
        return plotted, []

    detections = {
        "zhijin": detect_tissue(hsv, gray) if {"kuaizi", "shaozi"}.issubset(present) else None,
    }
    detections["yaqian"] = detect_toothpick(gray, [b for b in existing_boxes if b])

    extra_rows = []
    for label, box in detections.items():
        if label in present or box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(plotted, (x1, y1), (x2, y2), (255, 220, 40), 5)
        cv2.putText(
            plotted,
            f"OpenCV {label}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 220, 40),
            2,
            cv2.LINE_AA,
        )
        extra_rows.append({
            "name": f"{label}（OpenCV补偿）",
            "score": 0.68,
            "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
        })
    return plotted, extra_rows


def correct_cutlery_rows(rows):
    labels = [normalize_label(row.get("name", "")) for row in rows]
    present = set(labels)
    if "yaqian" in present:
        return rows, []
    if not {"kuaizi", "shaozi", "zhijin"}.issubset(present):
        return rows, []

    spoon_indices = [i for i, label in enumerate(labels) if label == "shaozi"]
    if len(spoon_indices) < 2:
        return rows, []

    # In the validation misses, a wrapped toothpick is detected as a second,
    # lower-confidence spoon. Correct only this duplicate pattern.
    correction_index = min(spoon_indices, key=lambda i: float(rows[i].get("score", 0)))
    if float(rows[correction_index].get("score", 0)) > 0.85:
        return rows, []

    corrected = dict(rows[correction_index])
    corrected["name"] = "yaqian（OpenCV修正）"
    rows[correction_index] = corrected
    return rows, [corrected]
