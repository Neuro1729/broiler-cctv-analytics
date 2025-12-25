## Bird Counting and Weight Estimation from CCTV Video

### Overview

This project implements a prototype system for automated **bird counting and weight estimation** from fixed-camera poultry CCTV footage. The system combines deep-learning-based object detection with multi-object tracking to produce:

* Bird counts over time (timestamp → count)
* Per-bird weight estimation using a **relative weight proxy**
* Annotated CCTV video with tracking IDs and bounding boxes
* A FastAPI service for automated batch or live video analysis

The solution is designed for commercial poultry farm environments where cameras are static and birds move naturally within a constrained area.

---

## Methodology

### 1. Bird Counting (Detection + Tracking)

**Detector**
A custom-trained **YOLO** model is used to detect individual birds in CCTV frames.

**Tracker**
**ByteTrack** (via BoxMOT) assigns persistent IDs to birds across frames.

Each bird receives a **stable track ID**, allowing us to:

* Count unique birds per frame
* Avoid double counting
* Follow the same bird over time

The count at any timestamp is simply the number of **active tracking IDs** in that frame.

---

### 2. Tiled Inference (High-Density Flocks)

Poultry CCTV often contains **dozens of birds in a single frame**, which reduces detection accuracy if processed at full resolution.

To solve this, we use **spatial tiling**:

* Each frame is split into a **3×3 grid (9 tiles)**
* YOLO runs on each tile independently
* Bounding boxes are merged back into full-frame coordinates

This improves:

* Small bird detection
* Occluded bird recovery
* Overall recall in dense scenes

---

### 3. Handling Occlusions and ID Stability

ByteTrack performs **motion-based association** with confidence-aware matching.
We configure a **large track buffer (e.g., 600 frames)** so that a bird’s ID is not lost when:

* It passes behind another bird
* It is temporarily missed by the detector
* Lighting or motion blur causes low confidence

Counting is based on **tracking IDs, not raw detections**, which prevents double-counting during overlaps.

---

## Weight Estimation

True bird weights are not available in the CCTV data, so the system uses a **relative weight proxy**.

### Weight Proxy (Implemented)

For each tracked bird:

```
weight_index = bounding_box_width × bounding_box_height   (pixels²)
```

In a fixed camera:

* Larger birds occupy more pixels
* Smaller birds occupy fewer pixels

Thus, bounding-box area is a **strong proxy for relative mass**.

Each bird’s area is logged per frame and stored in JSON along with detection confidence.

---

### Converting Proxy → Real Weight (grams)

To convert the weight index into real-world grams, the following is required:

1. **Camera calibration**

   * Pixel → centimeter conversion
2. **Ground-truth weights**

   * A small set of birds weighed manually
3. **Regression model**

   * Learn:

     ```
     weight (g) = m × area + c
     ```

Once calibrated, the same pipeline can output **true bird weight and flock biomass**.

---

## Video Preprocessing

To enable fast inference and deployment:

* Videos are converted to **H.264 MP4**
* Frame rate can be sampled (e.g., 10 FPS)
* Resolution can be downscaled to ~720p

This reduces storage by >90% while preserving tracking accuracy for slow-moving poultry.

---

## API

### Health Check

```
GET /health
```

Returns:

```
{"status": "OK"}
```

---

### Analyze Video

```
POST /analyze_video
```

Accepts a CCTV video and returns:

* `counts_over_time`
* `tracks_sample`
* `weight_estimates`
* `artifacts` (annotated video + JSON)

---

## Limitations and Assumptions

* Fixed camera (no pan/tilt)
* Birds are visible most of the time
* Weight estimates are **relative**, not absolute grams
* No long-term full occlusions

---

## Future Improvements

* Camera calibration for gram-accurate weights
* Temporal smoothing of weight curves
* Appearance-based re-identification (ReID embeddings)
* Flock-level biomass and growth tracking
* Live RTSP streaming support
