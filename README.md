
# 🐔 Bird Counting and Weight Estimation from Poultry CCTV Video

## Overview
This project implements a prototype system for **automated bird counting and weight estimation** from fixed-camera poultry CCTV footage. Leveraging **deep learning** and **multi-object tracking**, the system delivers:

- **Bird counts over time** (timestamp → count)  
- **Per-bird relative weight estimation** using bounding-box area as a proxy  
- **Annotated output video** with tracking IDs and bounding boxes  
- A **FastAPI service** for automated batch analysis of uploaded videos  

Designed for **commercial poultry farms** with static cameras and naturally moving birds in constrained spaces.

---
## Demo

[![Watch the video](https://img.youtube.com/vi/0BOnEA6IgQI/0.jpg)](https://youtu.be/0BOnEA6IgQI)

## 🧠 Methodology

### 1. **Bird Counting (Detection + Tracking)**
- **Detector**: Custom-trained **YOLOv8** model (`best.pt`) fine-tuned on poultry CCTV data  
- **Tracker**: **ByteTrack** (via `boxmot`) for robust, persistent ID assignment  
- Count per frame = number of **active unique track IDs** → avoids double-counting  

### 2. **Tiled Inference for Dense Flocks**
To handle frames with **dozens of birds**, each frame is split into a **3×3 grid**:
- YOLO runs independently on each tile  
- Detections are merged back to full-frame coordinates  
- **Improves recall** for small/occluded birds in crowded scenes  

### 3. **Occlusion Robustness & ID Stability**
- **Large track buffer** (600 frames) maintains IDs during:
  - Temporary occlusions
  - Missed detections (due to blur/lighting)
- Tracking (not raw detection) ensures **stable counts**  

### 4. **Weight Estimation (Relative Proxy)**
Since true weights are unavailable:
```python
weight_proxy = bounding_box_width × bounding_box_height  # (pixels²)
```
- In a **fixed camera**, pixel area correlates strongly with bird mass  
- Each tracked bird logs area + confidence per frame in JSON output  

> 🔧 **To convert to grams**: calibrate using known weights + camera pixel-to-cm ratio, then fit a linear regression:  
> `weight (g) = m × area + c`

### 5. **Video Preprocessing (For Efficiency)**
- Input videos should be **H.264 MP4**, **~720p**, **10 FPS**  
- Reduces file size by **>90%** with minimal impact on tracking accuracy  

---

## 🚀 API Endpoints

### ✅ Health Check
```http
GET /health
```
**Response**
```json
{"status": "ok"}
```

### 🎥 Analyze Video
```http
POST /analyze_video
```
**Form Data**
- `video_file`: MP4 video (multipart upload)
- `conf_thresh` (optional, default=0.6): YOLO confidence threshold

**Response JSON**
```json
{
  "message": "Processing complete",
  "video_path": "/tmp/.../tracked.mp4",
  "json_path": "/tmp/.../analysis.json",
  "frames": 1250
}
```

> 💡 The actual **annotated video** and **full analysis JSON** are saved to disk. In production, serve these as downloadable files or return URLs.

---

## ▶️ How to Run the Project

### 1️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
```
- **Linux/macOS**: `source venv/bin/activate`  
- **Windows (CMD)**: `venv\Scripts\activate`

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ **FFmpeg** must be installed on your system (for OpenCV video I/O).  
> - Ubuntu: `sudo apt install ffmpeg`  
> - Windows: Included with many Python distributions or install via [ffmpeg.org](https://ffmpeg.org)

### 3️⃣ Start the FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
On success:
```
Uvicorn running on http://127.0.0.1:8000
```

---

## 🧪 Test the API (Windows Example)

### 🩺 Health Check
```cmd
curl.exe http://127.0.0.1:8000/health
```
✅ Expected: `{"status":"ok"}`

### 🎥 Analyze a Video
**Multi-line (CMD):**
```cmd
curl.exe -X POST http://127.0.0.1:8000/analyze_video ^
-F "video_file=@Input_Videos/sample.mp4" ^
-F "conf_thresh=0.01"
```

**Single-line (CMD or PowerShell):**
```cmd
curl.exe -X POST http://127.0.0.1:8000/analyze_video -F "video_file=@Input_Videos/sample.mp4" -F "conf_thresh=0.01"
```

> ✅ **Windows Tips**:
> - Always use `curl.exe` (not `curl`)
> - Use `@` before file path
> - Use `^` for line breaks in CMD

---

## 📁 Output Structure
After `/analyze_video`, results are saved in a temporary directory (returned in response):

```
analysis.json
├── video_info: {filename, fps, resolution, total_frames}
├── weight_estimation: {method, note}
└── frame_logs: [
      {
        "frame_id": 0,
        "timestamp_sec": 0.0,
        "bird_count": 24,
        "birds": [
          {"track_id": 5, "bbox": [x1,y1,x2,y2], "area_px2": 12400, "weight_proxy": 12400},
          ...
        ]
      },
      ...
    ]
```

Annotated video: `tracked.mp4` (green boxes + no text overlay for clarity)

---

## ⚠️ Limitations & Assumptions
- **Fixed camera** (no pan/tilt/zoom)  
- Birds are **mostly visible** (no long-term full occlusions)  
- **Weight estimates are relative** (not in grams)  
- Assumes birds move **slowly** (suitable for 10 FPS sampling)

---

## 🚧 Future Improvements
- Camera calibration → **absolute weight (grams)**  
- Temporal smoothing of weight curves  
- Appearance-based Re-ID to reduce ID switches  
- Flock-level **biomass & growth tracking**  
- **RTSP/live stream** support

---

> 📦 **Demo outputs** (`tracked_output_ultra_small.mp4`, `bird_analysis.json`) are generated 
