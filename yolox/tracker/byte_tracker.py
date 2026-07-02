import numpy as np
from collections import deque

class STrack:
    def __init__(self, tlwh, score, track_id):
        self.tlwh = tlwh  # [x1, y1, x2, y2]
        self.score = score
        self.track_id = track_id

class BYTETracker:
    def __init__(self, track_thresh=0.4, match_thresh=0.8, track_buffer=30, frame_rate=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.next_id = 1
        self.tracks = []
        self.lost = deque(maxlen=track_buffer)

    def update(self, dets, img_height, img_width):
        # dets: Nx5 [x1, y1, x2, y2, score]
        updated_tracks = []
        for det in dets:
            x1, y1, x2, y2, score = det
            if score < self.track_thresh:
                continue
            found = False
            for t in self.tracks:
                iou = self._iou([x1, y1, x2, y2], t.tlwh)
                if iou > self.match_thresh:
                    t.tlwh = [x1, y1, x2, y2]
                    t.score = score
                    updated_tracks.append(t)
                    found = True
                    break
            if not found:
                new_track = STrack([x1, y1, x2, y2], score, self.next_id)
                self.next_id += 1
                updated_tracks.append(new_track)
        self.tracks = updated_tracks
        return self.tracks

    def _iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
