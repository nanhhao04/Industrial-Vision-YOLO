#  Industrial Assembly Line Inspection System

A high-performance, real-time computer vision system designed for multi-stage industrial assembly line inspection. This system utilizes YOLOv8 to detect components and slots, verifying correct assembly through advanced spatial logic and IoU-based matching.

##  Key Features

*   **Multi-Stage Inspection Pipeline:** Supports multiple production stages, each with its own dedicated camera and inspection criteria.
*   **Real-Time RTSP Processing:** High-performance RTSP reading with GStreamer integration (GPU-accelerated decoding via `nvh265dec` with CPU fallback).
*   **Intelligent Component Verification:** Implements IoU (Intersection over Union) matching logic to ensure components are correctly aligned with their designated slots.
*   **Automated Stage Transition:** Seamlessly switches between inspection stages once the current stage requirements are verified and stable.
*   **Dual-Model Architecture:** Uses specialized YOLO models for slot detection and component identification to maximize precision.
*   **Visual Feedback System:** Real-time visualization of detections, status counters, and transition overlays.

## 🏗 System Architecture

The system operates in a sequential stage-based workflow:

1.  **RTSP Stream Acquisition:** Threaded `RTSPReader` captures frames using the most efficient available backend (GStreamer UDP/TCP GPU -> GStreamer CPU -> FFmpeg).
2.  **Detection Layer:**
    *   **Model 1 (Slot):** Detects physical mounting slots on the assembly.
    *   **Model 2 (Component):** Detects the parts being assembled.
3.  **Logic Layer (Inspection):**
    *   Extracts bounding boxes for target components.
    *   Calculates IoU between components and slots.
    *   Verifies if the required number of matches is achieved (e.g., Stage 1 requires 3 matches for Component 3).
4.  **Stability & Transition:** Uses frame-based counters to ensure a "Pass" status is stable before automatically transitioning the camera feed and logic to the next stage.

##  Project Structure

```text
CuoiKyAI/
├── main.py              # Core application logic and pipeline
├── result/              # Model checkpoints
│   ├── Component/
│   │   └── Slot.pt      # YOLOv8 model for slot detection
│   └── Slot/
│       └── Component.pt # YOLOv8 model for component detection
└── result/              # (Optional) Inspection results and logs
```

## 🛠 Installation

### Prerequisites

*   **Python 3.8+**
*   **CUDA-enabled GPU** (Recommended for real-time performance)
*   **GStreamer** with `nvcodec` (for Jetson/NVIDIA hardware acceleration)

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/CuoiKyAI.git
    cd CuoiKyAI
    ```

2.  Install dependencies:
    ```bash
    pip install ultralytics opencv-python numpy
    ```

3.  (Optional) For GStreamer support on Ubuntu/Jetson:
    ```bash
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                         gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
                         gstreamer1.0-plugins-ugly gstreamer1.0-libav
    ```

## ⚙ Configuration

Modify the `STAGE_CONFIG` in `main.py` to match your hardware setup:

```python
STAGE_CONFIG = {
    1: {
        "rtsp": "rtsp://user:pass@IP:554/stream",
        "targets": [3],
        "desc": "Stage 1: Component 3"
    },
    # ... add more stages
}
```

##  Usage

Run the main inspection pipeline:

```bash
python main.py
```

### Controls
*   **`q`**: Quit the application.
*   **Window Resizing**: The viewer window is resizable (`1280x720` default).

##  Logic & Matching

The system uses a 0.5 IoU threshold to determine if a component is correctly seated in its slot.
*   **Standard Match:** At least one component matched to a slot.
*   **Special Rules:** Easily customizable logic per stage (e.g., verifying multiple identical components in a single view).

