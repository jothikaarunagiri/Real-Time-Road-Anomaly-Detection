import argparse
import time
import os
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort


# =============================
# CONFIG
# =============================
CLASS_NAMES = [
    "pothole",
    "barrier",
    "fallen_tree",
    "road_debris",
    "traffic_cone"
]

# BGR colors (no red used)
CLASS_COLORS = {
    "pothole": (0, 255, 0),          # Green
    "barrier": (255, 0, 0),          # Blue
    "fallen_tree": (255, 0, 255),    # Purple
    "road_debris": (0, 255, 255),    # Yellow
    "traffic_cone": (0, 165, 255)    # Orange
}

CONF_THRESHOLD_DEFAULT = 0.30
NMS_THRESHOLD = 0.35


# =============================
# Utility
# =============================
def letterbox(image, new_shape=320):
    h, w = image.shape[:2]
    scale = new_shape / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (nw, nh))

    canvas = np.zeros((new_shape, new_shape, 3), dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    return canvas, scale, top, left


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (180, 0, 255),  # Pink
        6,
        cv2.LINE_AA
    )


# =============================
# Main
# =============================
def main(opt):

    os.makedirs("logs", exist_ok=True)

    csv_name = f"logs/detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_name, "w") as f:
        f.write("timestamp,class,confidence,fps\n")

    session = ort.InferenceSession(opt.weights, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # -------- Camera / Video Handling --------
    if opt.source.isdigit():
        cap = cv2.VideoCapture(int(opt.source), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(opt.source)

    if not cap.isOpened():
        print("Cannot open source")
        return

    prev_time = time.time()

    cv2.namedWindow("Road Anomaly Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Road Anomaly Detection", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img, scale, pad_top, pad_left = letterbox(frame, opt.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        preds = session.run([output_name], {input_name: img})[0]

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        draw_fps(frame, fps)

        boxes, scores, class_ids = [], [], []

        for det in preds[0]:
            x, y, w, h = det[:4]
            obj_conf = float(det[4])
            class_scores = det[5:]

            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            conf = obj_conf * class_conf

            if conf < opt.conf_thres:
                continue

            x1 = int((x - w / 2 - pad_left) / scale)
            y1 = int((y - h / 2 - pad_top) / scale)
            x2 = int((x + w / 2 - pad_left) / scale)
            y2 = int((y + h / 2 - pad_top) / scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, opt.conf_thres, NMS_THRESHOLD)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = scores[i]
                class_name = CLASS_NAMES[class_ids[i]]

                color = CLASS_COLORS[class_name]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(
                    frame,
                    f"{class_name} {conf:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                with open(csv_name, "a") as f:
                    f.write(f"{ts},{class_name},{conf:.3f},{fps:.2f}\n")

        cv2.imshow("Road Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================
# Args
# =============================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True,
                        help="0 for USB cam or video file path")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf-thres", type=float, default=CONF_THRESHOLD_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_opt())
