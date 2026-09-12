import argparse
from pathlib import Path

from ultralytics import YOLO


def train(model_name, data, project, name, epochs, imgsz, batch, device, workers, task=None):
    model = YOLO(model_name)
    kwargs = {
        "data": str(Path(data).resolve()),
        "project": str(Path(project).resolve()),
        "name": name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "exist_ok": True,
    }
    if task:
        kwargs["task"] = task
    return model.train(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use 0 for the first CUDA GPU, cpu for CPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-cls", action="store_true")
    parser.add_argument("--skip-det", action="store_true")
    parser.add_argument("--skip-seg", action="store_true")
    args = parser.parse_args()

    datasets = Path(args.datasets)
    project = Path(args.project)
    if not args.skip_cls:
        train("yolov8n-cls.pt", datasets / "packaging_cls", project / "classify", "packaging", args.epochs, args.imgsz, args.batch, args.device, args.workers)
    if not args.skip_det:
        train("yolov8n.pt", datasets / "components_det" / "data.yaml", project / "detect", "components", args.epochs, args.imgsz, args.batch, args.device, args.workers)
    if not args.skip_seg:
        train("yolov8n-seg.pt", datasets / "cutlery_seg" / "data.yaml", project / "segment", "cutlery", args.epochs, args.imgsz, args.batch, args.device, args.workers)


if __name__ == "__main__":
    main()
