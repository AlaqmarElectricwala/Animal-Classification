# 🐾 Animal Image Classification System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-MobileNetV3--Large-blue)](https://pytorch.org/vision/stable/models/mobilenetv3.html)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning system for **19-species animal classification** using fine-tuned **MobileNetV3-Large** in PyTorch, paired with a modern, glassmorphism-styled **Streamlit** web interface.

---

## 🌟 Key Features

- **🚀 MobileNetV3-Large Backbone**: Optimized for lightweight, high-speed inference without compromising precision.
- **🎯 19 Animal Classes**: Classifies a diverse range of species from lions and leopards to koalas and wombats.
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
│   └── dataset/
│       └── README.md       # Dataset split metrics & breakdown
├── model/
│   └── animal_classification_mobilenetv3.pth  # Trained model weights checkpoint
└── README.md               # Main project documentation
```

---

## 📊 Dataset Breakdown

The dataset spans 19 animal species with split statistics as detailed below:

| Species | Train Set | Validation Set | Test Set | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **Badger** | 1,317 | 221 | 188 | 1,726 |
| **Bison** | 944 | 200 | 212 | 1,356 |
| **Boar** | 42 | 5 | 8 | 55 |
| **Cheetah** | 2,267 | 206 | 208 | 2,681 |
| **Chimpanzee** | 1,028 | 194 | 190 | 1,412 |
| **Cougar** | 963 | 175 | 191 | 1,329 |
| **Giraffe** | 889 | 114 | 103 | 1,106 |
| **Gorilla** | 951 | 225 | 208 | 1,384 |
| **Hippopotamus** | 1,005 | 213 | 223 | 1,441 |
| **Jaguar** | 1,151 | 16 | 20 | 1,187 |
| **Koala** | 3,464 | 177 | 238 | 3,879 |
| **Leopard** | 1,284 | 188 | 208 | 1,680 |
| **Lion** | 1,507 | 192 | 219 | 1,918 |
| **Llama** | 950 | 215 | 205 | 1,370 |
| **Orangutan** | 1,380 | 192 | 206 | 1,778 |
| **Snow Leopard** | 1,282 | 180 | 202 | 1,664 |
| **Tiger** | 2,492 | 179 | 202 | 2,873 |
| **Weasel** | 1,121 | 172 | 188 | 1,481 |
| **Wombat** | 1,272 | 217 | 184 | 1,673 |
| **Total** | **25,809** | **3,284** | **3,595** | **32,688** |

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
model/animal_classification_mobilenetv3.pth
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
- **Classifier Head**: Modified `Linear(1280, 19)` final projection layer
- **Preprocessing Pipeline**:
  - Image Resizing: Shortest edge scaled to `256px`
  - Center Crop: `224 × 224 px`
  - Normalization: ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Supported Formats**: JPEG (`.jpg`, `.jpeg`) and PNG (`.png`) up to **10 MB**

---

## 📜 License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
