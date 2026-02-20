# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Updated for event-based logging + big FPS display

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights,
    source,
    data,
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    view_img=False,
):

    # ------------------- Setup -------------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    model.warmup(imgsz=(1, 3, *imgsz))

    # ------------------- Logging -------------------
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "detections.csv"

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame", "class", "confidence", "fps"])

    # ------------------- Video Window -------------------
    window_name = "YOLOv5 Road Anomaly Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    prev_time = time.time()
    frame_id = 0

    # ------------------- Inference Loop -------------------
    for path, im, im0, vid_cap, s in dataset:
        frame_id += 1

        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # FPS calculation (real-time)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        annotator = Annotator(im0, line_width=3, example=str(names))

        # ------------------- BIG + BOLD FPS -------------------
        cv2.putText(
            im0,
            f"FPS: {fps:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.2,                 # BIG SIZE
            (180, 0, 255),       # PURPLE
            8,                   # BOLD
            cv2.LINE_AA,
        )

        # ------------------- Event-based detection -------------------
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # keep only highest-confidence detection per class
                best_per_class = {}

                for *xyxy, conf, cls in det:
                    cls = int(cls)
                    conf = float(conf)
                    if cls not in best_per_class or conf > best_per_class[cls][1]:
                        best_per_class[cls] = (xyxy, conf)

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                for cls, (xyxy, conf) in best_per_class.items():
                    label = f"{names[cls]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(cls, True))

                    # write ONE log entry per class per frame
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp,
                            frame_id,
                            names[cls],
                            f"{conf:.3f}",
                            f"{fps:.2f}",
                        ])

        # ------------------- Display -------------------
        if view_img:
            cv2.imshow(window_name, im0)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    opt = parser.parse_args()
    opt.imgsz = tuple(opt.imgsz)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
