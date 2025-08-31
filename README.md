# CSI5110 Project – Playing Cards Classifier

This is a computer vision project for classifying playing cards from images using deep learning. The project supports model training, quantization for edge deployment (e.g., on Xilinx Kria KV260), and a live demo using a webcam.

## GitHub Usernames

- **Pranav Sinha**: [pranavs1997](https://github.com/pranavs1997)  
- **Harshan Brar**: [S3AN-7](https://github.com/S3AN-7)  
- **Eduard Sufaj**: [EddS84](https://github.com/EddS84)

---

## Requirements

- Python 3.10  
- Linux (tested on Ubuntu 20.04)  
- Optional: NVIDIA GPU for training  
- Kaggle account with API token (for dataset)

---

## 🔧 Setup Instructions

### 1. Clone and enter the project directory:
```bash
git clone https://github.com/pranavs1997/csi5110-project.git
cd csi5110-project
```

### 2. Create a Python virtual environment:
```bash
python -m venv .venv
```

> 💡 If you get an error, install venv using:
```bash
sudo apt install python3.10-venv
```

### 3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 4. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📥 Download Dataset

Before running the scripts, place your Kaggle API token `kaggle.json` in:

```bash
~/.config/kaggle/kaggle.json
```

### 5. Run dataset download script:
```bash
python dataset.py
```

---

## 🧠 Train and Test the Model

### 6. Train the model:
```bash
python train.py
```

### 7. Test the model:
```bash
python test.py
```

---

## ⚙️ Quantization (For Edge Deployment)

### 1. Navigate to the quantize folder:
```bash
cd quantize
```

### 2. Download dataset again (if needed):
```bash
python dataset.py
```

### 3. Transfer the folder to your Vitis-AI environment, launch the container, and select `vitis-ai-tensorflow2`.

### 4. Uncomment lines 21–49 in `train.py`, then run:
```bash
python train.py
```

### 5. Compile the quantized model:
```bash
bash ./compile.sh
```

---

## 📷 Live Demo (Webcam Inference)

### 1. Navigate to the demo directory:
```bash
cd demo
```

### 2. Install camera dependencies:
```bash
bash ./camscript.sh
```

### 3. Capture an image using the webcam:
```bash
bash ./webjpg.sh
```

### 4. Run inference on captured image:
```bash
bash ./launch.sh
```

---

## 📁 Project Structure (Simplified)
```
csi5110-project/
├── data/                  # CSV and image dataset
├── demo/                  # Webcam inference scripts
├── logs/                  # Training logs and plots
├── quantize/              # Quantization, FPGA support
├── dataset.py             # Dataset utility
├── model.py               # CNN architecture
├── train.py, test.py      # Training and testing scripts
├── requirements.txt
└── README.md
```
---
## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
