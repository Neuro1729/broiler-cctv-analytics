import os
import cv2
import numpy as np
import tempfile
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from boxmot import ByteTrack

# =========================
# CONFIG
# =========================
MODEL_PATH = "best.pt"

TILE_ROWS = 3
TILE_COLS = 3

DEFAULT_CONF = 0.6
MATCH_THRESH = 0.8
DEFAULT_FPS = 30

# Weight proxy info (for JSON metadata)
WEIGHT_MODEL = {
    "method": "area_to_weight",
    "note": "weight_proxy = bounding_box_area (pixels^2). Must be calibrated to grams using real data."
}

# =========================
# INIT
# =========================
app = FastAPI(title="Chicken Tracking + Weight Estimation API")
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    conf_thresh: float = Form(DEFAULT_CONF)
):
    temp_dir = tempfile.mkdtemp()

    # Save uploaded video
    video_path = os.path.join(temp_dir, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JSONResponse({"error": "Invalid video"}, status_code=400)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(temp_dir, "tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Global ByteTrack
    tracker = ByteTrack(
        track_thresh=conf_thresh,
        match_thresh=MATCH_THRESH,
        frame_rate=fps
    )

    analysis_data = {
        "video_info": {
            "filename": video.filename,
            "fps": fps,
            "resolution": [width, height],
            "total_frames": total_frames
        },
        "weight_estimation": WEIGHT_MODEL,
        "frame_logs": []
    }

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tile_h = height // TILE_ROWS
        tile_w = width // TILE_COLS

        detections = []

        # -----------------------------
        # 1) 3x3 tiled YOLO
        # -----------------------------
        for i in range(TILE_ROWS):
            for j in range(TILE_COLS):
                y1 = i * tile_h
                y2 = height if i == TILE_ROWS - 1 else (i + 1) * tile_h
                x1 = j * tile_w
                x2 = width if j == TILE_COLS - 1 else (j + 1) * tile_w

                tile = frame[y1:y2, x1:x2]
                res = model(tile, conf=conf_thresh, verbose=False)[0]

                if res.boxes is None:
                    continue

                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()

                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1

                for b, c, cls in zip(boxes, confs, clss):
                    detections.append([*b, c, cls])

        dets = np.array(detections) if detections else np.empty((0, 6))

        # -----------------------------
        # 2) ByteTrack (global)
        # -----------------------------
        tracks = tracker.update(dets, frame)

        # -----------------------------
        # 3) Per-frame logging
        # -----------------------------
        timestamp_sec = frame_id / fps
        birds = []

        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t[:5])
            area_px = (x2 - x1) * (y2 - y1)

            birds.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "area_px2": int(area_px),
                "weight_proxy": int(area_px)
            })

            # Draw green box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        analysis_data["frame_logs"].append({
            "frame_id": frame_id,
            "timestamp_sec": round(timestamp_sec, 3),
            "bird_count": len(birds),
            "birds": birds
        })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    json_path = os.path.join(temp_dir, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis_data, f, indent=2)

    return JSONResponse({
        "message": "Processing complete",
        "video_path": out_path,
        "json_path": json_path,
        "frames": total_frames
    })
