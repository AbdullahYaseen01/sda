"""
precompute_refs_resnet.py - Precompute ResNet50 features for product images

This script walks through products/ directory and extracts deep learning features
from all product images using a pre-trained ResNet50, saving them to refs_resnet.pkl
for later use.

ResNet50 is pre-trained on ImageNet (1.2M images, 1000 classes) so it generalizes
well even with just 5 images per product.

Requirements:
    pip install torch torchvision pillow numpy pandas

Usage:
    python precompute_refs_resnet.py

Directory structure expected:
    products/
        <product_name>/
            image1.jpg
            image2.png
            ...
        <product_name>/
            ...
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Configuration constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRODUCTS_DIR = "products"
OUTPUT_FILE = "refs_resnet.pkl"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
FEATURE_DIM = 2048  # ResNet50 outputs 2048-dim features

# Image preprocessing - ResNet50 standard
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
])


def load_pretrained_resnet50():
    """Load pre-trained ResNet50 model and remove classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Remove the classification head to get feature extractor
    # ResNet50 outputs 2048-dim features from the avg pooling layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    # Move to device and set to eval mode
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    
    return feature_extractor


def extract_features(image_path: str, model: nn.Module) -> np.ndarray:
    """Extract ResNet50 features from an image."""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
        
        # Convert to numpy and flatten
        features = features.squeeze().cpu().numpy()
        
        # Normalize features to unit length for cosine similarity
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def walk_product_images(base_dir: str) -> Dict[str, List[str]]:
    """
    Walk through products directory and collect all product folders.
    Returns dict mapping product_name to list of image paths.
    """
    product_dict = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_dir}' does not exist!")
        return product_dict
    
    # Walk through each folder in products
    for product_folder in sorted(base_path.iterdir()):
        if not product_folder.is_dir():
            continue
        
        # Skip classes.txt if it exists
        if product_folder.name == 'classes.txt':
            continue
        
        product_name = product_folder.name
        image_paths = []
        
        # Collect all supported images in this folder
        for img_file in sorted(product_folder.iterdir()):
            if img_file.is_file() and img_file.suffix in SUPPORTED_EXTENSIONS:
                image_paths.append(img_file)
        
        if image_paths:
            product_dict[product_name] = sorted(image_paths)
            print(f"Found {len(image_paths)} images for product '{product_name}'")
    
    return product_dict


def main():
    """Main function to precompute ResNet50 features."""
    print("=" * 70)
    print("ResNet50 Feature Precomputation for Product Detection")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Load pre-trained ResNet50
    print("\nLoading pre-trained ResNet50 model...")
    model = load_pretrained_resnet50()
    print("[OK] Model loaded successfully")
    
    # Walk through product images
    print(f"\nScanning '{PRODUCTS_DIR}' directory...")
    product_dict = walk_product_images(PRODUCTS_DIR)
    
    if not product_dict:
        print(f"\nError: No product folders found in '{PRODUCTS_DIR}'!")
        print("Please ensure the directory exists and contains product folders.")
        return
    
    print(f"\n[OK] Found {len(product_dict)} products")
    
    # Extract features for all images
    print("\nExtracting ResNet50 features...")
    refs_data = {}
    total_images = 0
    
    for product_name, image_paths in sorted(product_dict.items()):
        print(f"\nProcessing '{product_name}'...")
        product_features = []
        successful = 0
        
        for img_path in image_paths:
            features = extract_features(str(img_path), model)
            if features is not None:
                product_features.append({
                    'image': str(img_path),
                    'features': features
                })
                successful += 1
                total_images += 1
        
        if product_features:
            refs_data[product_name] = product_features
            print(f"  [OK] Extracted features from {successful}/{len(image_paths)} images")
        else:
            print(f"  ✗ Failed to extract features from any images")
    
    if not refs_data:
        print("\nError: Could not extract features from any product images!")
        return
    
    # Save to pickle file
    print(f"\nSaving {total_images} feature vectors to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(refs_data, f)
    
    print("=" * 70)
    print(f"[OK] Successfully precomputed features for {len(refs_data)} products")
    print(f"[OK] Total images processed: {total_images}")
    print(f"[OK] Features saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
