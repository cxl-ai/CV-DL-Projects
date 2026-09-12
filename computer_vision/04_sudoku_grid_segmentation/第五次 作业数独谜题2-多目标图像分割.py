"""多目标数独图像分割（代码2 / 拓展版）。

运行示例：
    python "第五次 作业数独谜题2-多目标图像分割.py" sudoku2.png
结果默认保存到 output_sudoku2 目录。若没有提供命令行参数，则读取 sudoku2.png。
"""

from pathlib import Path
import sys
import cv2
import numpy as np


OUTPUT_SIDE = 900          # 每个透视矫正后数独图的边长
MIN_AREA_RATIO = 0.012     # 候选轮廓最小面积占整图的比例，可按图像调整
CELL_SIZE = 70             # 数字拼图中每一个单元格的边长


def save_image(path, image):
    """保存图片；imencode+tofile 兼容 Windows 下含中文的输出路径。"""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def preprocess_for_detection(gray):
    """突出网格线，使同一数独的线条连成一个外部轮廓。"""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5,
    )
    # 很轻的闭运算，连接因拍照、阴影而断开的边线；核不能过大，避免粘连相邻题目。
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def order_points(points):
    """返回左上、右上、右下、左下四点。"""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def valid_sudoku_contours(binary):
    """从所有外部轮廓中保留形状、面积都像数独的多个候选区域。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = binary.shape[0] * binary.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_AREA_RATIO:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # 轮廓不是四边形时，用最小外接旋转矩形继续尝试，增强对拍照倾斜图的适应性。
        points = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
        if len(points) != 4:
            continue
        corners = order_points(points)
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        ratio = ((top + bottom) / 2) / max((left + right) / 2, 1)
        if not 0.60 <= ratio <= 1.40:
            continue
        # 排除细长、破碎轮廓；数独外框通常具有较高填充率。
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.65:
            continue
        candidates.append((corners, area, contour))

    # 阅读顺序：上到下、左到右，命名 sudoku_01、sudoku_02 ... 稳定可复现。
    candidates.sort(key=lambda item: (np.mean(item[0][:, 1]), np.mean(item[0][:, 0])))
    return candidates


def warp_sudoku(gray, corners, side=OUTPUT_SIDE):
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(gray, matrix, (side, side), flags=cv2.INTER_CUBIC)


def grid_visualisation(square):
    view = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)
    side = square.shape[0]
    for i in range(10):
        p = round(i * side / 9)
        colour = (0, 0, 255) if i % 3 == 0 else (0, 180, 0)
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(view, (p, 0), (p, side - 1), colour, thickness)
        cv2.line(view, (0, p), (side - 1, p), colour, thickness)
    return view


def extract_one_digit(cell):
    """去掉格线，保留单元格中央的有效连通域；空格返回全黑图。"""
    h, w = cell.shape
    binary = cv2.adaptiveThreshold(cv2.GaussianBlur(cell, (5, 5), 0), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    # 裁掉边缘，格线没有机会被当成数字；保留内部约 76% 的区域。
    margin = max(3, int(min(h, w) * 0.12))
    inner = binary[margin:h-margin, margin:w-margin]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inner, 8)
    digit = np.zeros_like(inner)
    min_area = inner.size * 0.025
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        # 过滤微小噪点和从残余格线延伸进来的细长成分。
        if area >= min_area and bh >= inner.shape[0] * 0.18 and bw >= inner.shape[1] * 0.05:
            digit[labels == index] = 255
    if not np.any(digit):
        return np.zeros((CELL_SIZE, CELL_SIZE), np.uint8)
    ys, xs = np.where(digit > 0)
    digit = digit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min((CELL_SIZE - 12) / digit.shape[0], (CELL_SIZE - 12) / digit.shape[1])
    resized = cv2.resize(digit, (max(1, round(digit.shape[1] * scale)), max(1, round(digit.shape[0] * scale))))
    canvas = np.zeros((CELL_SIZE, CELL_SIZE), np.uint8)
    y = (CELL_SIZE - resized.shape[0]) // 2
    x = (CELL_SIZE - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def segment_digits(square):
    """等分 9×9 网格，并把 81 个数字单元格拼回一张结果图。"""
    side = square.shape[0]
    digits = []
    for row in range(9):
        for col in range(9):
            y0, y1 = round(row * side / 9), round((row + 1) * side / 9)
            x0, x1 = round(col * side / 9), round((col + 1) * side / 9)
            digits.append(extract_one_digit(square[y0:y1, x0:x1]))
    rows = [np.concatenate(digits[row * 9:(row + 1) * 9], axis=1) for row in range(9)]
    mosaic = np.concatenate(rows, axis=0)
    # 用细灰线保留 9×9 的空间结构，便于人工检查每个单元格的分割。
    for i in range(10):
        p = round(i * mosaic.shape[0] / 9)
        cv2.line(mosaic, (p, 0), (p, mosaic.shape[0] - 1), 100, 1)
        cv2.line(mosaic, (0, p), (mosaic.shape[1] - 1, p), 100, 1)
    return mosaic


def process_multi_sudoku(image_path, output_dir="output_sudoku2"):
    image_path = Path(image_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"无法读取图片：{image_path.resolve()}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    binary = preprocess_for_detection(gray)
    save_image(output / "01_detection_binary.png", binary)

    candidates = valid_sudoku_contours(binary)
    overview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for number, (corners, _, _) in enumerate(candidates, 1):
        polygon = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overview, [polygon], True, (0, 0, 255), 3)
        anchor = tuple(corners[0].astype(int))
        cv2.putText(overview, f"Sudoku {number}", (anchor[0], max(25, anchor[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        square = warp_sudoku(gray, corners)
        stem = f"sudoku_{number:02d}"
        save_image(output / f"{stem}_corrected.png", square)
        save_image(output / f"{stem}_grid.png", grid_visualisation(square))
        save_image(output / f"{stem}_digits.png", segment_digits(square))

    save_image(output / "00_detection_overview.png", overview)
    print(f"检测到 {len(candidates)} 个数独区域，结果已保存到：{output.resolve()}")
    return len(candidates)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "sudoku2.png"
    target = sys.argv[2] if len(sys.argv) > 2 else "output_sudoku2"
    process_multi_sudoku(source, target)
