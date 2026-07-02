from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

class CrowdByteTracker:
    def __init__(self, track_thresh=0.4, match_thresh=0.8, track_buffer=30, frame_rate=30):
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )

    def update(self, detections, img_info):
        # detections: Nx6 [x1, y1, x2, y2, score, class]
        # Only pass person detections
        if len(detections) == 0:
            return []
        # ByteTrack expects [x1, y1, x2, y2, score]
        dets = np.array([d[:5] for d in detections], dtype=np.float32)
        # Update tracker
        online_targets = self.tracker.update(dets, img_info['height'], img_info['width'])
        tracks = []
        for t in online_targets:
            x1, y1, x2, y2 = t.tlwh
            track_id = t.track_id
            conf = t.score
            tracks.append([x1, y1, x2, y2, track_id, conf])
        return tracks
