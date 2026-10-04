# Image to Text Reader (Offline ML)

A local Machine Learning optical character recognition utility that reads text from images and reconstructs clean, structured text without calling external cloud AI APIs.

## Architecture

The system executes entirely on the local CPU without network requests or API keys:

1. **Image Preprocessing**: Auto-contrast adjustment, Otsu binarization, and projection profile deskewing.
2. **Offline Neural OCR**: Local Tesseract LSTM (Long Short-Term Memory) neural network engine.
3. **Text Reconstruction**:
   - Reconnects hyphenated line-wraps (`docu- \n ment` -> `document`)
   - Merges soft line breaks into cohesive paragraphs while retaining lists
   - Normalizes spacing around punctuation
   - Resolves common character/digit confusions
   - Offline Norvig dictionary spell correction

## Requirements

Ensure Python packages are installed:
```bash
pip install -r requirements.txt
```

On macOS, Tesseract binary is located at `/opt/homebrew/bin/tesseract`.

## Usage

### 1. Python API

```python
from reader import ImageTextReader

reader = ImageTextReader()
result = reader.read("samples/book_page.png")

print(result["proper_text"])
print(f"Confidence: {result['confidence']}%")
```

Direct text helper:
```python
text = reader.read_to_text("samples/receipt.png")
print(text)
```

### 2. Command Line (CLI)

```bash
# Basic extraction
python3 reader.py samples/book_page.png

# Print raw OCR alongside formatted text
python3 reader.py samples/receipt.png --raw

# Save output directly to file
python3 reader.py samples/quote_card.png --save output.txt
```

### 3. Local Web Terminal (Bitcoin DeFi Aesthetic)

Run the local server:
```bash
python3 app.py
```
Open **http://127.0.0.1:8001** in your browser.

Features:
- **Bitcoin DeFi Design System**: True Void (`#030304`) background, Dark Matter (`#0F1115`) panels, Bitcoin Orange (`#F7931A`) and Digital Gold (`#FFD600`) luminescent energy and glowing pill CTAs.
- **Precision Typography**: Space Grotesk headings, Inter body, JetBrains Mono telemetry metrics.
- **Single-Screen Layout**: Zero-scroll dual-pane terminal with live telemetry and clipboard support (`Cmd+V` / `Ctrl+V`).
- **Theme Toggle**: Dark / Light mode toggle with persistence in `localStorage`.

### 4. Custom Scikit-Learn Character Model

To train and evaluate an isolated Scikit-Learn character classifier from scratch:
```bash
python3 custom_char_model.py
```

## Repository Structure

```
image-to-text-reader/
├── reader.py               # Python API & command line interface
├── ocr_engine.py           # Preprocessing & Tesseract LSTM inference
├── text_formatter.py       # Post-processing and text reconstruction
├── custom_char_model.py    # Trainable Scikit-Learn character classifier
├── app.py                  # Local FastAPI web server
├── static/
│   ├── index.html          # Web interface
│   ├── style.css           # Utilitarian CSS stylesheet
│   └── script.js           # Client-side logic
├── samples/                # Test sample images
├── requirements.txt        # Dependencies
└── README.md
```
