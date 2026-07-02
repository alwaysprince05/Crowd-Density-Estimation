import time
import cv2
import numpy as np

DENSITY_COLORS = {
    "LOW": (0, 255, 0),
    "MODERATE": (0, 255, 255),
    "HIGH": (0, 140, 255),
    "CRITICAL": (0, 0, 255),
}

class FPSCounter:
    def __init__(self, avg_over=30):
        self.avg_over = avg_over
        self.times = []
        self.last_time = None
        self.fps = 0.0

    def start(self):
        self.last_time = time.time()

    def stop(self):
        if self.last_time is None:
            return 0.0
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        self.times.append(elapsed)
        if len(self.times) > self.avg_over:
            self.times.pop(0)
        self.fps = len(self.times) / sum(self.times) if self.times else 0.0
        return elapsed * 1000  # ms

    def get_fps(self):
        return self.fps

def draw_overlays(
    frame, tracks, people_count, density_level, fps, frame_time_ms
):
    color = DENSITY_COLORS.get(density_level, (255, 255, 255))
    # Draw tracks
    for track in tracks:
        x1, y1, x2, y2, track_id, conf = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}"
        cv2.putText(
            frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    # Overlay stats
    h, w = frame.shape[:2]
    stats_bg = np.zeros((90, w, 3), dtype=np.uint8)
    cv2.putText(
        stats_bg,
        f"People: {people_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
    )
    cv2.putText(
        stats_bg,
        f"Density: {density_level}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
    )
    cv2.putText(
        stats_bg,
        f"FPS: {fps:.1f} | Frame: {frame_time_ms:.1f} ms",
        (w - 350, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    frame[0 : stats_bg.shape[0], :, :] = cv2.addWeighted(
        frame[0 : stats_bg.shape[0], :, :], 0.5, stats_bg, 0.5, 0
    )
    return frame
