# 🐾 Animal Image Classifier — MobileNetV3

A production-quality **Streamlit** web application that classifies images of animals into **64 species** using a fine-tuned **MobileNetV3-Large** PyTorch model.

## Supported Animals

alpaca · angora · antelope · armadillo · badger · bat · bear · beetle · bison · boar · bullfrog · camel · capybara · cat · cheetah · chimpanzee · cock · colobus · cougar · cow · crocodile · deer · dog · elephant · fly · fox · giraffe · goldfish · gorilla · hammerhead · hamster · hare · hippopotamus · hog · hyena · indri · jaguar · koala · langur · leatherback turtle · leopard · lion · llama · marmoset · mink · mosquito · okapi · orangutan · possum · proboscis monkey · puffer · scorpion · seahorse · snake · snow leopard · squirrel · tarantula · tiger · tortoise · warthog · weasel · wolf spider · wombat · zebra

## Project Structure

```
├── model/
│   └── animal_classification_mobilenetv3_large.pth       # Trained checkpoint
├── app/
│   ├── app.py              # Streamlit UI entrypoint
│   ├── model.py            # Model architecture & checkpoint loading
│   ├── inference.py        # Preprocessing & prediction logic
│   ├── config.py           # Constants, paths, device selection
│   ├── requirements.txt    # Pinned dependencies
│   └── README.md           # This file
```

## Setup

### 1. Install dependencies

```bash
pip install -r app/requirements.txt
```

> **Note:** For CPU-only inference (no GPU), install PyTorch with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. Place the model checkpoint

Ensure the trained model file exists at:

```
model/animal_classification_mobilenetv3_large.pth
```

This should be a **MobileNetV3-Large** state_dict with a 64-class classifier head.

### 3. Run the app

From the **project root** directory:

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Open the app in your browser.
2. Upload a `.jpg`, `.jpeg`, or `.png` image (max 10 MB).
3. The model classifies the animal and displays:
   - Predicted species with confidence percentage
   - Top-5 predictions as a visual bar chart

## Expected Input

- **Format:** JPEG or PNG image files
- **Content:** Photographs of animals from the 64 supported species
- **Size:** Up to 10 MB per image
- **Preprocessing:** Images are automatically resized to 256px (shortest edge), center-cropped to 224×224, and normalised using ImageNet statistics

## Technical Details

| Component        | Detail                              |
|------------------|-------------------------------------|
| Architecture     | MobileNetV3-Large                   |
| Input size       | 224 × 224 px                        |
| Normalisation    | ImageNet mean/std                   |
| Output classes   | 64                                  |
| Inference device | Auto-detected (CUDA → MPS → CPU)   |
| Framework        | PyTorch + torchvision               |
| UI               | Streamlit                           |

## License

This project is provided as-is for demonstration and educational purposes.
