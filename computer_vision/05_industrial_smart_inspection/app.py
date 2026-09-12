import csv
import json
import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO

from scripts.opencv_cutlery import compensate_cutlery, correct_cutlery_rows

try:
    import torch
except Exception:
    torch = None


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "static" / "uploads"
RESULT_DIR = APP_ROOT / "static" / "results"
CONFIG_PATH = APP_ROOT / "configs" / "dataset_config.json"

TASKS = {
    "packaging": {
        "title": "包装 OK/NG 分类",
        "subtitle": "绿豆糕包装破损二分类",
        "model": APP_ROOT / "runs" / "classify" / "packaging" / "weights" / "best.pt",
        "results": APP_ROOT / "runs" / "classify" / "packaging" / "results.csv",
        "kind": "classify",
        "metric_keys": [
            ("准确率", "metrics/accuracy_top1"),
            ("召回率", "metrics/recall"),
            ("mAP50", "metrics/mAP50"),
            ("验证损失", "val/loss"),
        ],
    },
    "components": {
        "title": "电子元器件检测",
        "subtitle": "稳压管缺陷与电阻/电容识别",
        "model": APP_ROOT / "runs" / "detect" / "components" / "weights" / "best.pt",
        "results": APP_ROOT / "runs" / "detect" / "components" / "results.csv",
        "kind": "detect",
        "metric_keys": [("精确率", "metrics/precision(B)"), ("召回率", "metrics/recall(B)"), ("mAP50", "metrics/mAP50(B)")],
    },
    "cutlery": {
        "title": "餐具四件套分割",
        "subtitle": "筷子、勺子、牙签、纸巾实例分割",
        "model": APP_ROOT / "runs" / "segment" / "cutlery" / "weights" / "best.pt",
        "results": APP_ROOT / "runs" / "segment" / "cutlery" / "results.csv",
        "kind": "segment",
        "metric_keys": [("Mask 精确率", "metrics/precision(M)"), ("Mask 召回率", "metrics/recall(M)"), ("Mask mAP50", "metrics/mAP50(M)")],
    },
}


app = Flask(__name__)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def allowed_file(filename):
    return Path(filename).suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}


def fmt_metric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number <= 1:
        return f"{number * 100:.1f}%"
    return f"{number:.3f}"


def read_last_metrics(task):
    path = task["results"]
    if not path.exists():
        return {"epochs": 0, "items": []}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"epochs": 0, "items": []}
    last = rows[-1]
    return {
        "epochs": len(rows),
        "items": [{"label": label, "value": fmt_metric(last.get(key))} for label, key in task["metric_keys"]],
    }


def dataset_counts():
    return {
        "packaging": {
            "train": len(list((APP_ROOT / "datasets" / "packaging_cls" / "train").glob("*/*.jpg"))),
            "val": len(list((APP_ROOT / "datasets" / "packaging_cls" / "val").glob("*/*.jpg"))),
        },
        "components": {
            "train": len(list((APP_ROOT / "datasets" / "components_det" / "images" / "train").glob("*.jpg"))),
            "val": len(list((APP_ROOT / "datasets" / "components_det" / "images" / "val").glob("*.jpg"))),
        },
        "cutlery": {
            "train": len(list((APP_ROOT / "datasets" / "cutlery_seg" / "images" / "train").glob("*.jpg"))),
            "val": len(list((APP_ROOT / "datasets" / "cutlery_seg" / "images" / "val").glob("*.jpg"))),
        },
    }


def task_cards():
    counts = dataset_counts()
    cards = []
    for key, task in TASKS.items():
        cards.append({
            "key": key,
            "title": task["title"],
            "subtitle": task["subtitle"],
            "ready": task["model"].exists(),
            "epochs": read_last_metrics(task)["epochs"],
            "metrics": read_last_metrics(task)["items"],
            "train_count": counts[key]["train"],
            "val_count": counts[key]["val"],
        })
    return cards


def format_result(task_key, result):
    if task_key == "packaging":
        probs = result.probs
        if probs is None:
            return []
        names = result.names
        return [{"name": names[int(idx)], "score": float(probs.data[int(idx)])} for idx in probs.top5]

    rows = []
    boxes = result.boxes
    if boxes is None:
        return rows
    for box in boxes:
        cls_id = int(box.cls[0])
        rows.append({
            "name": result.names[cls_id],
            "score": float(box.conf[0]),
            "box": [round(float(x), 1) for x in box.xyxy[0].tolist()],
        })
    return rows


def run_prediction(task_key, image_path):
    task = TASKS[task_key]
    model_path = task["model"]
    if not model_path.exists():
        return None, [], f"模型文件不存在：{model_path}。请先运行 python scripts\\train_all.py。"

    model = YOLO(str(model_path))
    run_name = f"{task_key}_{uuid.uuid4().hex[:8]}"
    results = model.predict(
        str(image_path),
        conf=0.15 if task_key == "cutlery" else 0.25,
        device=os.getenv("YOLO_DEVICE", "0" if torch is not None and torch.cuda.is_available() else "cpu"),
        save=True,
        project=str(RESULT_DIR),
        name=run_name,
        exist_ok=True,
    )
    result = results[0]
    plotted = result.plot()
    rows = format_result(task_key, result)
    if task_key == "cutlery":
        rows, _ = correct_cutlery_rows(rows)
        plotted, extra_rows = compensate_cutlery(image_path, rows, plotted)
        rows.extend(extra_rows)
    result_file = RESULT_DIR / run_name / f"{image_path.stem}_result.jpg"
    import cv2

    cv2.imwrite(str(result_file), plotted)
    return result_file.relative_to(APP_ROOT).as_posix(), rows, None


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "tasks": TASKS,
        "task_cards": task_cards(),
        "active_task": "packaging",
        "uploaded": None,
        "result_image": None,
        "rows": [],
        "error": None,
        "config": load_config(),
        "cuda_ready": torch is not None and torch.cuda.is_available(),
    }
    if request.method == "POST":
        task_key = request.form.get("task", "packaging")
        context["active_task"] = task_key
        file = request.files.get("image")
        if task_key not in TASKS:
            context["error"] = "未知任务类型。"
        elif not file or file.filename == "":
            context["error"] = "请先选择一张图片。"
        elif not allowed_file(file.filename):
            context["error"] = "仅支持 bmp、jpg、jpeg、png 图片。"
        else:
            ext = Path(file.filename).suffix.lower()
            upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
            file.save(upload_path)
            context["uploaded"] = upload_path.relative_to(APP_ROOT).as_posix()
            result_image, rows, error = run_prediction(task_key, upload_path)
            context["result_image"] = result_image
            context["rows"] = rows
            context["error"] = error
    return render_template("index.html", **context)


@app.route("/samples/<path:filename>")
def samples(filename):
    return send_from_directory(APP_ROOT, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
