import cv2
import time
import threading
from ultralytics import YOLO


# ===========================================
# === GSTREAMER RTSP READER CONFIGURATION ===
# ===========================================

GST_WIDTH = 1920
GST_HEIGHT = 1080
GST_LATENCY = 200
GST_MAX_BUFFERS = 3
GST_DROP = True


class GStreamerPipeline:
    @staticmethod
    def build(url, protocol="udp"):
        if protocol == "udp":
            return (
                f"rtspsrc location={url} latency={GST_LATENCY} protocols=udp "
                f"udp-buffer-size=1048576 retry=5 timeout=5000000 "
                f"! rtph265depay ! h265parse ! "
                f"nvh265dec max-display-delay=0 ! "
                f"videoscale method=1 ! video/x-raw, width={GST_WIDTH}, height={GST_HEIGHT} ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink sync=false max-buffers={GST_MAX_BUFFERS} "
                f"drop={str(GST_DROP).lower()}"
            )
        else:
            return (
                f"rtspsrc location={url} latency={GST_LATENCY} protocols=tcp "
                f"tcp-timeout=5000000 retry=5 "
                f"! rtph265depay ! h265parse ! "
                f"nvh265dec max-display-delay=0 ! "
                f"videoscale method=1 ! video/x-raw, width={GST_WIDTH}, height={GST_HEIGHT} ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink sync=false max-buffers={GST_MAX_BUFFERS} "
                f"drop={str(GST_DROP).lower()}"
            )

    @staticmethod
    def build_cpu(url, protocol="udp"):
        """Fallback pipeline using CPU decoder (avdec_h265 or decodebin)"""
        if protocol == "udp":
            return (
                f"rtspsrc location={url} latency={GST_LATENCY} protocols=udp "
                f"udp-buffer-size=1048576 retry=5 timeout=5000000 "
                f"! rtph265depay ! h265parse ! "
                f"avdec_h265 ! " # Fallback to CPU decoder
                f"videoscale method=1 ! video/x-raw, width={GST_WIDTH}, height={GST_HEIGHT} ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink sync=false max-buffers={GST_MAX_BUFFERS} "
                f"drop={str(GST_DROP).lower()}"
            )
        else:
            return (
                f"rtspsrc location={url} latency={GST_LATENCY} protocols=tcp "
                f"tcp-timeout=5000000 retry=5 "
                f"! rtph265depay ! h265parse ! "
                f"avdec_h265 ! " # Fallback to CPU decoder
                f"videoscale method=1 ! video/x-raw, width={GST_WIDTH}, height={GST_HEIGHT} ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink sync=false max-buffers={GST_MAX_BUFFERS} "
                f"drop={str(GST_DROP).lower()}"
            )


class RTSPReader(threading.Thread):
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def run(self):
        # 1. Try GStreamer (GPU - nvh265dec) - UDP
        print("🔄 Trying GStreamer (UDP + GPU)...")
        gst = GStreamerPipeline.build(self.url, protocol="udp")
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            # 2. Try GStreamer (GPU - nvh265dec) - TCP
            print("⚠️ UDP GPU failed. Trying GStreamer (TCP + GPU)...")
            gst = GStreamerPipeline.build(self.url, protocol="tcp")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            # 3. Try GStreamer (CPU - avdec_h265) - UDP
            print("⚠️ GPU pipeline failed. Trying GStreamer (UDP + CPU)...")
            gst = GStreamerPipeline.build_cpu(self.url, protocol="udp")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            # 4. Try GStreamer (CPU - avdec_h265) - TCP
            print("⚠️ UDP CPU failed. Trying GStreamer (TCP + CPU)...")
            gst = GStreamerPipeline.build_cpu(self.url, protocol="tcp")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            # 5. Fallback to default OpenCV backend (FFmpeg)
            print("⚠️ GStreamer failed. Trying standard OpenCV VideoCapture (FFmpeg)...")
            cap = cv2.VideoCapture(self.url)

        if not cap.isOpened():
            print("❌ Camera cannot open RTSP stream (All methods failed)")
            return

        print("✅ RTSP Reader Started")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            with self.lock:
                self.frame = frame.copy()

        cap.release()
        print("🛑 RTSP Reader Stopped")


# ==========================
# === IOU + DRAW HELPERS ===
# ==========================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else (inter / union)


def normalize_name(name):
    return name.lower().replace("_", "").replace(" ", "")

def extract_boxes_by_name(result, target_name):
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes

    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    target_norm = normalize_name(target_name)

    for bb, c in zip(xyxy, cls):
        raw_name = names.get(int(c), str(c))
        if normalize_name(raw_name) == target_norm:
            boxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
    return boxes


def draw_boxes(result, img, color=(0, 255, 0), thickness=2):
    if result.boxes is None or len(result.boxes) == 0:
        return img

    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()

    for bb, c, cf in zip(xyxy, cls, conf):
        x1, y1, x2, y2 = map(int, bb.tolist())
        label = f"{names.get(int(c), c)} {cf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def put_status_top_right(img, text, color):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 3

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 10
    x = max(0, w - tw - pad)
    y = pad + th + 5

    cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)
    return img


def best_iou_to_any(slot_box, comp_boxes):
    best = 0.0
    for c in comp_boxes:
        best = max(best, iou_xyxy(slot_box, c))
    return best


# =====================
# === MAIN PROGRAM ====
# =====================
if __name__ == "__main__":

    # Model 1: Detect slots
    model1 = YOLO("result/Component/Slot.pt")

    # Model 2: Detect components
    model2 = YOLO("result/Slot/Component.pt")

    CONF1, IOU1 = 0.55, 0.55
    CONF2, IOU2 = 0.55, 0.55

    # --- STAGE CONFIGURATION ---
    # Configure RTSP URLs for each stage's camera
    STAGE_CONFIG = {
        1: {
            "rtsp": "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",  # Stage 1 Camera
            "targets": [3],                                            # Component 3
            "desc": "Stage 1: Component 3"
        },
        2: {
            "rtsp": "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",  # Stage 2 Camera
            "targets": [1, 2],                                         # Component 1 & 2
            "desc": "Stage 2: Component 1 & 2"
        },
        3: {
            "rtsp": "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",  # Stage 3 Camera
            "targets": [4, 5, 6],                                      # Component 4, 5 & 6
            "desc": "Stage 3: Component 4, 5 & 6"
        },
        4: {
            "rtsp": "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main",  # Stage 4 Camera
            "targets": [7, 8],                                         # Component 7 & 8
            "desc": "Stage 4: Component 7 & 8"
        }
    }

    current_stage = 1
    reader = None
    
    def start_camera(stage_num):
        url = STAGE_CONFIG[stage_num]["rtsp"]
        new_reader = RTSPReader(url)
        new_reader.start()
        print(f"🎥 Switched to Camera for Stage {stage_num}: {url}")
        return new_reader

    # Start initial camera
    reader = start_camera(current_stage)

    is_image = False 
    cap = None

    # Create a resizable window
    cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Viewer", 1280, 720)

    FRAME_SKIP = 5
    frame_count = 0
    REQUIRED_FRAMES_OK = 5
    
    status_counters = {i: 0 for i in range(1, 9)}
    
    stage_complete_timer = 0
    STAGE_TRANSITION_DELAY = 60 # Frames to show "Stage Complete" before switching (approx 2-3s)

    stage_stable_counter = 0
    STAGE_STABLE_DELAY = 10 # Frames required to be stable before completing (approx 0.5s)

    # Initialize variables for smooth inference
    res1 = None
    res2 = None
    
    while True:
        if not is_image:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1

        else:
            pass

        # If we are in a transition delay (Stage Completed)
        if stage_complete_timer > 0:
            stage_complete_timer -= 1
            
            # Draw Transition Screen Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            msg = f"STAGE {current_stage} COMPLETE!"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            cx, cy = frame.shape[1]//2, frame.shape[0]//2
            cv2.putText(frame, msg, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            cv2.imshow("Viewer", frame)
            cv2.waitKey(1)
            
            if stage_complete_timer == 0:
                # Transition now
                if current_stage < 4:
                    print(f"🚀 Moving to Stage {current_stage + 1}")
                    current_stage += 1
                    
                    # Stop old reader
                    reader.stop_event.set()
                    # Wait a bit?
                    time.sleep(0.5)
                    # Start new reader
                    reader = start_camera(current_stage)
                    
                    status_counters = {i: 0 for i in range(1, 9)}
                    # Reset result cache for new stage to avoid ghost boxes
                    res1 = None
                    res2 = None
                else:
                    print("🎉 ALL STAGES COMPLETED!")
            continue


        # --- 1. INFERENCE & LOGIC UPDATE (Only every FRAME_SKIP frames) ---
        if frame_count % FRAME_SKIP == 0 or res1 is None:
            res1 = model1.predict(frame, conf=CONF1, iou=IOU1, verbose=False)[0]  # slots
            res2 = model2.predict(frame, conf=CONF2, iou=IOU2, verbose=False)[0]  # components

            # Get targets for current stage
            current_targets = STAGE_CONFIG[current_stage]["targets"]
            
            all_targets_ok = True

            for i in current_targets:
                slot_name = f"Slot_{i}"
                comp_name = f"Component_{i}"

                slots = extract_boxes_by_name(res1, slot_name)
                comps = extract_boxes_by_name(res2, comp_name)

                is_pair_ok = False

                # --- INSPECTION LOGIC ---
                if i == 3:
                    # Stage 1 Special: Requires 3 matches for Component 3
                    if len(comps) == 3 and len(slots) > 0:
                        matches = 0
                        for c in comps:
                            best_iou = best_iou_to_any(c, slots)
                            if best_iou > 0.5: 
                                matches += 1
                        
                        if matches == 3:
                            is_pair_ok = True
                else:
                    # Standard Logic
                    if len(comps) >= 1 and len(slots) > 0:
                            max_match_iou = 0
                            for c in comps:
                                iou = best_iou_to_any(c, slots)
                                if iou > max_match_iou:
                                    max_match_iou = iou
                            
                            if max_match_iou > 0.5:
                                is_pair_ok = True

                # --- UPDATE STATUS ---
                if is_pair_ok:
                    status_counters[i] += 1
                else:
                    status_counters[i] = 0
                
                # Check individual target status for stage completion
                if status_counters[i] < REQUIRED_FRAMES_OK:
                    all_targets_ok = False
                else:
                     # Cap the counter to prevent overflow (optional)
                     status_counters[i] = min(status_counters[i], REQUIRED_FRAMES_OK + 10)

            # CHECK STAGE COMPLETION (Stability Check)
            if all_targets_ok:
                stage_stable_counter += 1
            else:
                stage_stable_counter = 0

            if stage_stable_counter >= STAGE_STABLE_DELAY:
                stage_complete_timer = STAGE_TRANSITION_DELAY
                stage_stable_counter = 0 # Reset to avoid repeated triggering


        # --- 2. VISUALIZATION (Every Frame) ---
        vis = frame.copy()
        
        if res1 is not None:
            vis = draw_boxes(res1, vis, color=(255, 255, 0), thickness=2)
        if res2 is not None:
            vis = draw_boxes(res2, vis, color=(255, 0, 255), thickness=2)

        stage_desc = STAGE_CONFIG[current_stage]["desc"]
        current_targets = STAGE_CONFIG[current_stage]["targets"] # Re-fetch just in case
        
        # Display Stage Info
        cv2.putText(vis, stage_desc, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Display Target Status
        for i in current_targets:
            final_status = "FALSE"
            final_color = (0, 0, 255) # Red
            
            if status_counters[i] >= REQUIRED_FRAMES_OK:
                final_status = "OK"
                final_color = (0, 255, 0) # Green

            label_text = f"Comp {i}: {final_status} ({status_counters[i]})"
            
            # Configurable position for list
            list_start_y = 80
            idx_in_list = current_targets.index(i)
            cy = list_start_y + (idx_in_list * 40)
            
            cv2.putText(vis, label_text, (30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2)

        cv2.imshow("Viewer", vis)

        key = cv2.waitKey(0 if is_image else 1) & 0xFF
        if key == ord("q"):
            break

    if reader:
        reader.stop_event.set()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
