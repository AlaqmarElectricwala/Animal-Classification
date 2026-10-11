# 🐾 Animal Image Classification System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-MobileNetV3--Large-blue)](https://pytorch.org/vision/stable/models/mobilenetv3.html)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning system for **93-species animal classification** using a fine-tuned **MobileNetV3-Large** PyTorch architecture, paired with a modern, glassmorphism-styled **Streamlit** web interface.

---

## 🌟 Key Features

- **🚀 MobileNetV3-Large Backbone**: Optimized for lightweight, high-speed inference without compromising precision.
- **🎯 93 Animal Classes**: Trained to recognize a diverse taxonomy of 93 species—spanning mammals, birds, reptiles, amphibians, insects, arachnids, and marine life.
- **🎨 Glassmorphism Streamlit UI**: Dark mode interactive web application featuring real-time top-5 probability distribution charts, animal chips, and responsive layout.
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
│   ├── requirements.txt    # App-specific Python dependencies
│   └── README.md           # App-specific documentation
├── data/
│   ├── dataset/            # Dataset splits (train, val, test) across 94 classes
│   └── README.md           # Full dataset split metrics & per-species breakdown
├── model/
│   └── animal_classification_mobilenetv3_large.pth  # Trained model checkpoint (93 classes)
├── train.ipynb             # Training, evaluation & checkpoint export notebook
├── requirements.txt        # Full project dependencies (app + training pipeline)
└── README.md               # Main project documentation
```

---

## 📊 Dataset Overview

The dataset contains **139,208 total images** organized across **94 animal species** in standardized ImageFolder splits:

| Split | Image Count | Percentage |
| :--- | :---: | :---: |
| **Train Set** | 108,016 | 77.6% |
| **Validation Set** | 15,456 | 11.1% |
| **Test Set** | 15,736 | 11.3% |
| **Total** | **139,208** | **100.0%** |

> 📌 **Note on Classes**: The currently deployed model checkpoint classifies across **93 species**. A complete per-class tabular breakdown across all 94 species is available in [`data/README.md`](data/README.md).

---

## 🐾 Supported Animal Species (93)

<details>
<summary>Click to view all 93 supported animal species</summary>

Alpaca · Angora · Antelope · Armadillo · Badger · Barn Spider · Bat · Bear · Bee · Beetle · Bison · Boar · Bullfrog · Camel · Capybara · Cat · Caterpillar · Cheetah · Chimpanzee · Cock · Cockroach · Colobus · Cougar · Cow · Coyote · Crocodile · Crow · Deer · Dog · Dolphin · Donkey · Dragonfly · Duck · Eagle · Elephant · Flamingo · Fly · Fox · Gazelle · Giraffe · Goat · Goldfish · Gorilla · Grasshopper · Guinea Pig · Hammerhead · Hamster · Hare · Hippopotamus · Hog · Hyena · Indri · Jaguar · Jellyfish · Kangaroo · Koala · Ladybugs · Langur · Leatherback Turtle · Leopard · Lion · Llama · Marmoset · Mink · Mongoose · Mosquito · Mule · Okapi · Orangutan · Porcupine · Possum · Proboscis Monkey · Puffer · Red Panda · Reindeer · Rock Crab · Scorpion · Seahorse · Seal · Snail · Snake · Snow Leopard · Spider Monkey · Squirrel · Tarantula · Terrapin · Tiger · Tortoise · Warthog · Weasel · Wolf Spider · Wombat · Zebra

</details>

---

## ⚙️ Quick Start

### 1. Prerequisites
Ensure **Python 3.9+** is installed on your machine.

### 2. Install Dependencies

You can install all project dependencies (inference + training tools):

```bash
pip install -r requirements.txt
```

Or install only the lightweight dependencies required to run the Streamlit app:

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
model/animal_classification_mobilenetv3_large.pth
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
- **Classifier Head**: Modified `Linear(1280, 93)` final projection layer
- **Preprocessing Pipeline**:
  - Image Resizing: Shortest edge scaled to `256px`
  - Center Crop: `224 × 224 px`
  - Normalization: ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Supported Formats**: JPEG (`.jpg`, `.jpeg`) and PNG (`.png`) up to **10 MB**
- **Inference Latency**: Real-time (~10–30ms on GPU, ~50–120ms on CPU)

---

## 📜 License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
