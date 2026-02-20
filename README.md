🚦 Real-Time Road Anomaly Detection from Dashcam Footage
📌 Overview

This project implements a real-time road anomaly detection system using deep learning and edge deployment techniques. The system processes live dashcam footage and detects critical road hazards to enhance road safety.

The detection model is exported in ONNX format (best.onnx) for optimized inference.

🎯 Detected Road Obstacles

The system detects:

🕳 Potholes

🚧 Road Barriers

🪵 Road Debris

🚦 Traffic Cones

⚠ Other unexpected obstacles

🧠 Model Details

Architecture: YOLO-based object detection

Format: ONNX (best.onnx)

Task: Real-time object detection

Deployment: Edge device (Raspberry Pi / Local system)

The model is optimized for efficient inference on resource-constrained hardware.

🛠 Technologies Used

Python

OpenCV

ONNX Runtime

YOLO framework

Git & GitHub

📂 Project Structure
edge_ai_road_detection/
│
├── detect_pi.py        # Main inference script
├── best.onnx           # Exported ONNX model
├── yolov5/             # YOLO framework (submodule)
├── README.md
└── .gitignore
▶️ Setup Instructions
1️⃣ Clone the Repository (Important for submodule)
git clone --recurse-submodules <your-repo-link>

If already cloned:

git submodule update --init --recursive
2️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt is not present, install manually:

pip install opencv-python onnxruntime numpy
3️⃣ Run Detection
python detect_pi.py

The system will:

Capture video feed

Perform inference using best.onnx

Draw bounding boxes

Display detected anomalies in real time

🚀 Key Features

Real-time detection

Edge deployment ready

Lightweight ONNX inference

Modular architecture

Scalable for smart transportation systems

📈 Future Improvements

Speed-based alert triggering

Audio/visual warning system

GPS tagging of anomalies

Cloud-based monitoring dashboard

Hardware acceleration (Edge TPU / NPU)
