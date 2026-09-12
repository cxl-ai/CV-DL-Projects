import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gbk"))


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def inner_data_dir(folder: Path) -> Path:
    children = [p for p in folder.iterdir() if p.is_dir()]
    return children[0] if len(children) == 1 else folder


def collect_groups(root: Path):
    groups = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if folder.name in {"datasets", "scripts", "configs", "templates", "static", "runs"}:
            continue
        data_dir = inner_data_dir(folder)
        images = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        jsons = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".json"])
        if images:
            groups.append({"name": folder.name, "dir": data_dir, "images": images, "jsons": jsons})
    return groups


def shape_types(group):
    types = set()
    labels = set()
    for jp in group["jsons"]:
        for shape in load_json(jp).get("shapes", []):
            types.add(shape.get("shape_type") or "")
            labels.add(shape.get("label") or "")
    return types, labels


def detect_task_groups(groups):
    tasks = {"packaging": [], "components": [], "cutlery": []}
    for group in groups:
        types, labels = shape_types(group)
        if not types:
            tasks["packaging"].append(group)
        elif types == {"rectangle"}:
            tasks["components"].append(group)
        elif "polygon" in types:
            tasks["cutlery"].append(group)
    return tasks


def write_jpg(src: Path, dst: Path):
    img = cv2.imdecode(np.frombuffer(src.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"Cannot encode image: {src}")
    dst.write_bytes(encoded.tobytes())
    return img.shape[1], img.shape[0]


def split_items(items, val_count, seed):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = min(val_count, max(1, len(items) - 1))
    return items[val_count:], items[:val_count]


def packaging_label_from_json(json_path: Path, class_names, flag_label_map):
    if not json_path.exists():
        return None
    flags = load_json(json_path).get("flags") or {}
    enabled = [name for name in class_names if flags.get(name) is True]
    mapped = [label for flag, label in flag_label_map.items() if flags.get(flag) is True]
    candidates = enabled + mapped
    candidates = [label for label in candidates if label in class_names]
    return candidates[0] if len(candidates) == 1 else None


def load_packaging_csv(root: Path, csv_path: str):
    path = root / csv_path
    if not path.exists():
        return {}
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = (row.get("image") or "").replace("\\", "/").strip()
            label = (row.get("label") or "").strip()
            if image and label:
                labels[image] = label
    return labels


def write_packaging_template(root: Path, csv_path: str, unlabeled):
    path = root / csv_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for img in unlabeled:
            writer.writerow([img.relative_to(root).as_posix(), ""])


def prepare_packaging(root: Path, out: Path, groups, cfg, seed):
    class_names = cfg["class_names"]
    csv_labels = load_packaging_csv(root, cfg.get("label_csv", "configs/packaging_labels.csv"))
    flag_label_map = cfg.get("flag_label_map", {})
    val_per_class = int(cfg.get("val_per_class", 5))
    by_class = {name: [] for name in class_names}
    unlabeled = []
    for group in groups:
        for image in group["images"]:
            rel = image.relative_to(root).as_posix()
            label = csv_labels.get(rel)
            if not label:
                label = packaging_label_from_json(image.with_suffix(".json"), class_names, flag_label_map)
            if label in by_class:
                by_class[label].append(image)
            else:
                unlabeled.append(image)

    if unlabeled:
        write_packaging_template(root, cfg.get("label_csv", "configs/packaging_labels.csv"), unlabeled)
        raise ValueError(
            "Packaging images need image-level OK/NG labels. "
            f"Fill {cfg.get('label_csv', 'configs/packaging_labels.csv')} with OK or NG, then rerun."
        )

    for class_name, images in by_class.items():
        train, val = split_items(images, val_per_class, seed)
        for split, split_images in [("train", train), ("val", val)]:
            for src in split_images:
                dst = out / split / class_name / f"{src.parent.parent.name}_{src.stem}.jpg"
                write_jpg(src, dst)


def yolo_box(points, width, height):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(0, min(xs)), min(width, max(xs))
    y1, y2 = max(0, min(ys)), min(height, max(ys))
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def prepare_detection(root: Path, out: Path, groups, cfg, seed):
    names = cfg["class_names"]
    class_to_id = {name: i for i, name in enumerate(names)}
    pairs = []
    for group in groups:
        for img in group["images"]:
            jp = img.with_suffix(".json")
            if jp.exists():
                pairs.append((img, jp))
    train, val = split_items(pairs, int(cfg.get("val_images", 15)), seed)
    for split, split_pairs in [("train", train), ("val", val)]:
        for img, jp in split_pairs:
            stem = f"{img.parent.parent.name}_{img.stem}"
            width, height = write_jpg(img, out / "images" / split / f"{stem}.jpg")
            lines = []
            for shape in load_json(jp).get("shapes", []):
                label = shape.get("label")
                if label not in class_to_id:
                    continue
                cx, cy, bw, bh = yolo_box(shape.get("points", []), width, height)
                if bw > 0 and bh > 0:
                    lines.append(f"{class_to_id[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_path = out / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines), encoding="utf-8")
    write_dataset_yaml(out, names)


def prepare_segmentation(root: Path, out: Path, groups, cfg, seed):
    names = cfg["class_names"]
    class_to_id = {name: i for i, name in enumerate(names)}
    pairs = []
    for group in groups:
        for img in group["images"]:
            jp = img.with_suffix(".json")
            if jp.exists():
                pairs.append((img, jp))
    train, val = split_items(pairs, int(cfg.get("val_images", 20)), seed)
    for split, split_pairs in [("train", train), ("val", val)]:
        for img, jp in split_pairs:
            stem = f"{img.parent.parent.name}_{img.stem}"
            width, height = write_jpg(img, out / "images" / split / f"{stem}.jpg")
            lines = []
            for shape in load_json(jp).get("shapes", []):
                label = shape.get("label")
                points = shape.get("points", [])
                if label not in class_to_id or len(points) < 3:
                    continue
                coords = []
                for x, y in points:
                    coords.extend([min(max(x / width, 0), 1), min(max(y / height, 0), 1)])
                line = " ".join([str(class_to_id[label])] + [f"{v:.6f}" for v in coords])
                lines.append(line)
            label_path = out / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines), encoding="utf-8")
    write_dataset_yaml(out, names)


def write_dataset_yaml(out: Path, names):
    yaml_text = [
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_text.extend([f"  {i}: {name}" for i, name in enumerate(names)])
    (out / "data.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")


def write_summary(out: Path, tasks):
    lines = ["# Dataset Preparation Summary", ""]
    for task, groups in tasks.items():
        lines.append(f"## {task}")
        for group in groups:
            types, labels = shape_types(group)
            lines.append(
                f"- {group['name']}: images={len(group['images'])}, json={len(group['jsons'])}, "
                f"shape_types={sorted(types)}, labels={sorted(labels)}"
            )
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Workspace root containing raw folders.")
    parser.add_argument("--config", default="configs/dataset_config.json")
    parser.add_argument("--out", default="datasets")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    out = root / args.out
    ensure_clean_dir(out)

    groups = collect_groups(root)
    tasks = detect_task_groups(groups)
    seed = int(cfg.get("seed", 42))

    prepare_packaging(root, out / "packaging_cls", tasks["packaging"], cfg["packaging"], seed)
    prepare_detection(root, out / "components_det", tasks["components"], cfg["components"], seed)
    prepare_segmentation(root, out / "cutlery_seg", tasks["cutlery"], cfg["cutlery"], seed)
    write_summary(out, tasks)
    print(f"Prepared datasets at {out}")
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
