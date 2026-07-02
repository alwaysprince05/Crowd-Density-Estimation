
import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO
from density import calculate_density, classify_density
from tracker import CrowdByteTracker
from utils import FPSCounter, draw_overlays
import importlib

def safe_yolo_load(model_path):
    # Patch for PyTorch 2.6+ safe loading
    DetectionModel = None
    Sequential = None
    Conv = None
    try:
        DetectionModel = getattr(importlib.import_module("ultralytics.nn.tasks"), "DetectionModel")
    except Exception:
        pass
    try:
        Sequential = getattr(importlib.import_module("torch.nn.modules.container"), "Sequential")
    except Exception:
        pass
    try:
        Conv = getattr(importlib.import_module("ultralytics.nn.modules"), "Conv")
    except Exception:
        pass
    safe_classes = []
    if DetectionModel is not None:
        safe_classes.append(DetectionModel)
    if Sequential is not None:
        safe_classes.append(Sequential)
    if Conv is not None:
        safe_classes.append(Conv)
    if safe_classes and hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals(safe_classes):
            return YOLO(model_path)
    else:
        return YOLO(model_path)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    use_half = config.get("use_half", True) and torch.cuda.is_available()
    model = safe_yolo_load(config.get("yolo_model", "yolov8n.pt"))
    if use_half:
        model.model.half()
    model.to(device)
    tracker = CrowdByteTracker(frame_rate=30)
    cap = cv2.VideoCapture(config["video_path"])
    if not cap.isOpened():
        print(f"Failed to open video: {config['video_path']}")
        return
    w, h = config["resize_width"], config["resize_height"]
    display = config.get("display", True)
    save_output = config.get("save_output", False)
    output_path = config.get("output_path", "output.mp4")
    out_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    fps_counter = FPSCounter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))
        fps_counter.start()
        # YOLOv8 inference
        results = model.predict(
            frame,
            device=device,
            half=use_half,
            classes=[0],  # person class only
            conf=0.3,
            verbose=False,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    dets.append([x1, y1, x2, y2, conf, 0])
        img_info = {"height": h, "width": w}
        tracks = tracker.update(dets, img_info)
        people_count = len(tracks)
        density_score = calculate_density(people_count)
        density_level = classify_density(density_score)
        frame_time_ms = fps_counter.stop()
        fps = fps_counter.get_fps()
        frame = draw_overlays(
            frame, tracks, people_count, density_level, fps, frame_time_ms
        )
        if display:
            cv2.imshow("Crowd Density Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if save_output and out_writer is not None:
            out_writer.write(frame)
    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
