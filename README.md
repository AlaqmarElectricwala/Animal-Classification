# 🐾 Animal Image Classification System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-MobileNetV3--Large-blue)](https://pytorch.org/vision/stable/models/mobilenetv3.html)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning system for **23-species animal classification** using fine-tuned **MobileNetV3-Large** in PyTorch, paired with a modern, glassmorphism-styled **Streamlit** web interface.

---

## 🌟 Key Features

- **🚀 MobileNetV3-Large Backbone**: Optimized for lightweight, high-speed inference without compromising precision.
- **🎯 23 Animal Classes**: Classifies a diverse range of species from domestic animals (cat, dog, cow) to wild species (lion, tiger, antelope, cheetah).
- **🎨 Glassmorphism Streamlit UI**: Dark mode interactive web application featuring real-time top-5 probability distribution charts.
- **⚡ Automatic Device Acceleration**: Seamless auto-detection and execution across **CUDA**, **Apple Silicon (MPS)**, and **CPU**.
- **🛡️ Production-Ready Pipeline**: Includes input validation, image corruption safety checks, file size constraints, and modular pure-function design.

---

## 📁 Directory Structure

```
Animal-Classification/
├── app/
│   ├── app.py              # Streamlit UI application entrypoint
│   ├── config.py           # Centralized configuration, constants, & device selection
│   ├── inference.py        # Image preprocessing and pure inference pipeline
│   ├── model.py            # MobileNetV3-Large architecture & checkpoint loader
│   ├── requirements.txt    # Pinned Python dependencies
│   └── README.md           # App-specific documentation
├── data/
│   ├── dataset/            # Dataset splits (train, val, test)
│   └── README.md           # Dataset split metrics & breakdown
├── model/
│   └── animal_classification_mobilenetv3_23classes.pth  # Trained model weights checkpoint
└── README.md               # Main project documentation
```

---

## 📊 Dataset Breakdown

The dataset spans 23 animal species with split statistics as detailed below:

| Species | Train Set | Validation Set | Test Set | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **Antelope** | 296 | 30 | 30 | 356 |
| **Badger** | 1,719 | 221 | 188 | 2,128 |
| **Bison** | 1,364 | 200 | 211 | 1,775 |
| **Boar** | 411 | 26 | 27 | 464 |
| **Cat** | 4,377 | 500 | 500 | 5,377 |
| **Cheetah** | 2,267 | 206 | 208 | 2,681 |
| **Chimpanzee** | 1,383 | 194 | 187 | 1,764 |
| **Cougar** | 963 | 175 | 191 | 1,329 |
| **Cow** | 360 | 30 | 30 | 420 |
| **Dog** | 4,375 | 500 | 500 | 5,375 |
| **Giraffe** | 1,285 | 114 | 103 | 1,502 |
| **Gorilla** | 1,332 | 225 | 208 | 1,765 |
| **Hippopotamus** | 1,403 | 213 | 223 | 1,839 |
| **Jaguar** | 1,151 | 16 | 20 | 1,187 |
| **Koala** | 3,857 | 177 | 238 | 4,272 |
| **Leopard** | 1,713 | 188 | 208 | 2,109 |
| **Lion** | 1,917 | 192 | 219 | 2,328 |
| **Llama** | 950 | 215 | 208 | 1,373 |
| **Orangutan** | 1,803 | 192 | 206 | 2,201 |
| **Snow Leopard** | 1,282 | 180 | 202 | 1,664 |
| **Tiger** | 2,894 | 179 | 202 | 3,275 |
| **Weasel** | 1,121 | 172 | 188 | 1,481 |
| **Wombat** | 1,685 | 217 | 184 | 2,086 |
| **Total** | **39,908** | **4,362** | **4,481** | **48,751** |

---

## ⚙️ Quick Start

### 1. Prerequisites
Ensure Python 3.9+ is installed on your machine.

### 2. Install Dependencies
Install the required libraries listed in `app/requirements.txt`:

```bash
pip install -r app/requirements.txt
```

> 💡 **CPU-Only PyTorch Installation**: If running on a CPU-only environment without CUDA, install PyTorch with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 3. Verify Model Checkpoint
Ensure the trained PyTorch weights file is placed at:
```
model/animal_classification_mobilenetv3_23classes.pth
```

### 4. Launch the Web App
Run Streamlit from the **root directory**:

```bash
streamlit run app/app.py
```

The web dashboard will launch automatically at `http://localhost:8501`.

---

## 🛠️ Technical Specifications

- **Model Architecture**: `MobileNetV3-Large` (`torchvision.models.mobilenet_v3_large`)
- **Classifier Head**: Modified `Linear(1280, 23)` final projection layer
- **Preprocessing Pipeline**:
  - Image Resizing: Shortest edge scaled to `256px`
  - Center Crop: `224 × 224 px`
  - Normalization: ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Supported Formats**: JPEG (`.jpg`, `.jpeg`) and PNG (`.png`) up to **10 MB**

---

## 📜 License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
