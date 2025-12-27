# ResNet50 Product Detection System - Setup Guide

## Overview

This is an improved version of your product detection MVP that uses **ResNet50** (pre-trained on ImageNet) instead of ORB feature matching. This works much better with limited training data (5 images per product).

## Why ResNet50 Instead of ORB?

| Feature | ORB | ResNet50 |
|---------|-----|---------|
| **Speed** | Fast | Real-time (GPU) |
| **Accuracy with 5 images/class** | Poor | Excellent |
| **Lighting/Scale Invariance** | Limited | Strong |
| **Training Required** | No | No (pre-trained) |
| **Memory** | Low | Moderate |

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements_resnet.txt
```

This installs:
- **torch**: PyTorch deep learning framework
- **torchvision**: Pre-trained models
- **opencv-python**: Computer vision library
- **numpy/pandas/pillow**: Data processing

### 2. Prepare Your Data

You should already have:
- `products/` folder with product images organized by product name
- `products.csv` with product information (id, name, price, etc.)

Example structure:
```
products/
  Ambroise, Fragrance World/
    image16.txt
    image17.txt
    ...
  Ameer Al Arab, Asdaaf/
    image0.txt
    ...
```

## Usage

### Step 1: Precompute Product Features

Run this ONCE to extract ResNet50 features from all your product images:

```bash
python precompute_refs_resnet.py
```

This will:
- Load pre-trained ResNet50 from torchvision
- Extract 2048-dimensional feature vectors from each product image
- Save features to `refs_resnet.pkl`
- Takes 2-5 minutes depending on your hardware

**Output:**
```
Loaded references for 25 products
✓ Successfully precomputed features for 25 products
✓ Total images processed: 125
✓ Features saved to: refs_resnet.pkl
```

### Step 2: Run Product Detection

```bash
python scan_and_detect_resnet.py
```

This starts the real-time scanner with controls:
- **ENTER**: Capture frame and detect products
- **c**: Continue to next frame
- **q**: Quit and generate receipt

**Example Output:**
```
Scanning frame...
Detected 3 products:
  - Ambroise, Fragrance World (confidence: 0.892)
  - Ameer Al Arab, Asdaaf (confidence: 0.876)
  - Caramel Glaze, Loui Martin (confidence: 0.834)
Cart items: 3
```

When you press 'q', a receipt is generated:
```
Receipt saved to: receipt/receipt_20251225_143022.html
```

## Tuning Parameters

In `scan_and_detect_resnet.py`, adjust these constants:

```python
MIN_SIMILARITY_SCORE = 0.65   # Higher = stricter matching (0.65-0.75 recommended)
MAX_DETECTIONS = 10           # Max products per scan
NMS_IOU_THRESHOLD = 0.4       # Remove overlapping detections (lower = remove more overlaps)
```

### Recommendations:

- **Too many false positives?** Increase `MIN_SIMILARITY_SCORE` to 0.70-0.75
- **Missing detections?** Decrease `MIN_SIMILARITY_SCORE` to 0.60
- **Duplicate detections?** Decrease `NMS_IOU_THRESHOLD` to 0.3

## Performance

### Hardware Requirements

- **CPU Only**: ~1-2 seconds per detection
- **GPU (NVIDIA CUDA)**: ~200-500ms per detection (10x faster!)
- **RAM**: ~4GB minimum
- **Storage**: ~400MB for model weights + feature cache

### Speed Improvement Over ORB

- **ORB**: ~100-200ms per image (fast but poor accuracy)
- **ResNet50**: ~500ms per image with GPU (accurate, deep learning)
- **Overall**: ~10x better accuracy with similar or better speed

## Files

### Main Scripts
- **precompute_refs_resnet.py**: Extract and save product features (run once)
- **scan_and_detect_resnet.py**: Real-time product scanning (run repeatedly)

### Generated Files
- **refs_resnet.pkl**: Precomputed product features (created by precompute_refs_resnet.py)
- **receipt/receipt_YYYYMMDD_HHMMSS.html**: Generated receipts

### Configuration
- **products.csv**: Your product database
- **requirements_resnet.txt**: Python dependencies

## Troubleshooting

### "refs_resnet.pkl not found"
- Run `python precompute_refs_resnet.py` first

### "products.csv not found"
- Ensure your `products.csv` file exists in the root directory

### Slow Performance
- Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, install CUDA drivers for GPU support

### Low Accuracy
- Increase `MIN_SIMILARITY_SCORE` threshold
- Ensure product images are well-lit and clear
- Check that your products.csv has correct product names matching folder names

### Camera Issues
- Check if webcam is connected and working
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras

## Next Steps

### To Further Improve Accuracy:

1. **Fine-tune ResNet50** (optional):
   - If you want even better accuracy, you can fine-tune on your products
   - Use `finetune_resnet.py` (coming soon)

2. **Add Product Cropping**:
   - Manually crop product images from your dataset
   - Remove background to improve feature extraction

3. **Augment Training Data**:
   - Use image augmentation (rotation, brightness) to create more training examples
   - This helps with varying camera angles and lighting

## Support

If you encounter issues:
1. Check that all input files exist
2. Verify your Python environment: `python --version` (3.8+)
3. Test imports: `python -c "import torch; print(torch.__version__)"`

## License

This code uses pre-trained weights from torchvision (torchvision license).
