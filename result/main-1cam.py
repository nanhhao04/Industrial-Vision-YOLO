import cv2
import time
import threading
from ultralytics import YOLO


# =========================================
# === GSTREAMER RTSP READER (UPDATED ONLY) ==
# =========================================

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

    # model1: detect slots
    model1 = YOLO("slot-full.pt")

    # model2: detect components
    model2 = YOLO("component-full.pt")

    CONF1, IOU1 = 0.55, 0.55
    CONF2, IOU2 = 0.55, 0.55

    RTSP_URL = "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main"
    reader = RTSPReader(RTSP_URL)
    reader.start()

    # VIDEO_PATH = "test/frame_0013_13s.jpg" # Đổi đường dẫn file video hoặc ảnh vào đây
    # VIDEO_PATH = "test_image.jpg" 

    # Check extension
    # is_image = VIDEO_PATH.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    is_image = False # RTSP is always video stream

    # if is_image:
    #     cap = None
    #     frame = cv2.imread(VIDEO_PATH)
    #     if frame is None:
    #         print(f"❌ Cannot open image: {VIDEO_PATH}")
    #         exit()
    # else:
    #     cap = cv2.VideoCapture(VIDEO_PATH)
    cap = None

    FRAME_SKIP = 5
    frame_count = 0

    out = None

    REQUIRED_FRAMES_OK = 1 if is_image else 5

    status_counters = {i: 0 for i in range(1, 9)}

    while True:
        if not is_image:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
        else:
            pass

        res1 = model1.predict(frame, conf=CONF1, iou=IOU1, verbose=False)[0]  # slots
        res2 = model2.predict(frame, conf=CONF2, iou=IOU2, verbose=False)[0]  # components

        vis = frame.copy()
        # Slots: Cyan (255, 255, 0) - Hi-tech look
        vis = draw_boxes(res1, vis, color=(255, 255, 0), thickness=2)
        # Components: Magenta (255, 0, 255) - High contrast
        vis = draw_boxes(res2, vis, color=(255, 0, 255), thickness=2)

        # Duyệt qua các cặp từ 1 đến 8
        for i in range(1, 9):
            slot_name = f"Slot_{i}"
            comp_name = f"Component_{i}"

            slots = extract_boxes_by_name(res1, slot_name)
            comps = extract_boxes_by_name(res2, comp_name)

            is_pair_ok = False

            # --- LOGIC KIỂM TRA ---
            if i == 3:
                if len(comps) == 3 and len(slots) > 0:
                    matches = 0
                    for c in comps:
                        best_iou = best_iou_to_any(c, slots)
                        if best_iou > 0.5: 
                            matches += 1
                    
                    if matches == 3:
                        is_pair_ok = True
            else:
                if len(comps) >= 1 and len(slots) > 0:
                     max_match_iou = 0
                     for c in comps:
                        iou = best_iou_to_any(c, slots)
                        if iou > max_match_iou:
                            max_match_iou = iou
                     
                     if max_match_iou > 0.5:
                         is_pair_ok = True

            # --- CẬP NHẬT TRẠNG THÁI ---
            if is_pair_ok:
                status_counters[i] += 1
            else:
                status_counters[i] = 0

            # --- HIỂN THỊ ---
            final_status = "FALSE"
            final_color = (0, 0, 255) # Red
            
            if status_counters[i] >= REQUIRED_FRAMES_OK:
                final_status = "OK"
                final_color = (0, 255, 0) # Green
                status_counters[i] = min(status_counters[i], REQUIRED_FRAMES_OK + 10)

            # Debug info: Show IoU/Count
            debug_info = f"Cnt:{status_counters[i]}"
            
            if i == 3:
                matches = 0
                if len(comps) == 3 and len(slots) > 0:
                     for c in comps:
                        if best_iou_to_any(c, slots) > 0.5: matches += 1
                debug_info += f" M:{matches}/3"
            elif len(comps) > 0 and len(slots) > 0:
                 max_iou = 0
                 for c in comps:
                    iou = best_iou_to_any(c, slots)
                    if iou > max_iou: max_iou = iou
                 debug_info += f" IoU:{max_iou:.2f}"

            label = f"{i}: {final_status} ({debug_info})"
            
            # Tọa độ vẽ list
            cx = 30
            cy = 30 + (i * 40) 
            
            cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2)


        cv2.imshow("Viewer", vis)

        key = cv2.waitKey(0 if is_image else 1) & 0xFF
        if key == ord("q"):
            break

    reader.stop_event.set()
    # if out is not None:
    #     out.release()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
