import cv2
import torch
import numpy as np
import yaml
from flask import Flask, render_template, Response, jsonify, request
from density import calculate_density, classify_density
from tracker import CrowdByteTracker
from utils import FPSCounter, draw_overlays
import importlib
import time
import threading

app = Flask(__name__)

# System State
state_lock = threading.Lock()
latest_frame = None
is_running = True
conf_threshold = 0.3
should_reset = False

live_stats = {
    "people_count": 0,
    "density_level": "LOW",
    "fps": 0.0,
    "latency": 0.0,
    "config": {}
}

def safe_yolo_load(model_path):
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
            from ultralytics import YOLO
            return YOLO(model_path)
    else:
        from ultralytics import YOLO
        return YOLO(model_path)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Background processing thread
def video_processing_thread():
    global latest_frame, live_stats, is_running, conf_threshold, should_reset
    
    config = load_config()
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    use_half = config.get("use_half", True) and torch.cuda.is_available()
    yolo_model_path = config.get("yolo_model", "yolov8n.pt")
    video_path = config["video_path"]
    w, h = config["resize_width"], config["resize_height"]
    
    with state_lock:
        live_stats["config"] = {
            "yolo_model": yolo_model_path,
            "device": device,
            "resize_width": w,
            "resize_height": h,
            "conf_threshold": conf_threshold
        }

    # Load YOLO Model
    model = safe_yolo_load(yolo_model_path)
    if use_half:
        model.model.half()
    model.to(device)
    
    tracker = CrowdByteTracker(frame_rate=30)
    fps_counter = FPSCounter()
    
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            time.sleep(2)
            continue
            
        while True:
            # Check for reset request
            if should_reset:
                with state_lock:
                    should_reset = False
                tracker = CrowdByteTracker(frame_rate=30)
                break
                
            if not is_running:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                # Video ended, restart
                break
                
            frame = cv2.resize(frame, (w, h))
            fps_counter.start()
            
            # YOLOv8 inference with dynamic conf_threshold
            results = model.predict(
                frame,
                device=device,
                half=use_half,
                classes=[0],  # person class only
                conf=conf_threshold,
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
            
            # Update stats
            with state_lock:
                live_stats["people_count"] = people_count
                live_stats["density_level"] = density_level
                live_stats["fps"] = fps
                live_stats["latency"] = frame_time_ms
                live_stats["config"]["conf_threshold"] = conf_threshold
            
            # Draw overlays
            frame = draw_overlays(
                frame, tracks, people_count, density_level, fps, frame_time_ms
            )
            
            # Encode frame to JPEG
            ret_encoded, jpeg_buffer = cv2.imencode('.jpg', frame)
            if ret_encoded:
                with state_lock:
                    latest_frame = jpeg_buffer.tobytes()
                    
            # Regulate processing speed (FPS)
            time.sleep(0.01)
            
        cap.release()

def stream_generator():
    global latest_frame
    while True:
        with state_lock:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        # Regulate stream feeding to browser client
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with state_lock:
        return jsonify(live_stats)

@app.route('/control', methods=['POST'])
def control():
    global is_running, should_reset
    action = request.json.get('action')
    with state_lock:
        if action == 'play':
            is_running = True
        elif action == 'pause':
            is_running = False
        elif action == 'reset':
            should_reset = True
            is_running = True
    return jsonify({"status": "success", "is_running": is_running})

@app.route('/config', methods=['POST'])
def update_config():
    global conf_threshold
    val = request.json.get('conf_threshold')
    if val is not None:
        try:
            with state_lock:
                conf_threshold = float(val)
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid confidence threshold"}), 400
    return jsonify({"status": "success", "conf_threshold": conf_threshold})

if __name__ == '__main__':
    # Start the worker thread for background processing
    worker = threading.Thread(target=video_processing_thread, daemon=True)
    worker.start()
    
    app.run(host='127.0.0.1', port=5001, debug=False)
