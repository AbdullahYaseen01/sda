"""
precompute_refs.py - Precompute ORB features for product images

This script walks through product_images/ directory and computes ORB keypoints
and descriptors for all product images, saving them to refs_orb.pkl for later use.

Requirements:
    pip install opencv-contrib-python numpy pandas

Usage:
    python precompute_refs.py

Directory structure expected:
    product_images/
        <id1>/
            image1.jpg
            image2.png
            ...
        <id2>/
            ...
"""

import os
import cv2
import pickle
import numpy as np
from pathlib import Path

# Configuration constants
ORB_NFEATURES = 1500  # Number of ORB features to detect
PRODUCT_IMAGES_DIR = "product_images"
OUTPUT_FILE = "refs_orb.pkl"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def load_image(path):
    """Load an image from file path."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def compute_orb_features(image, orb):
    """Compute ORB keypoints and descriptors for an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def walk_product_images(base_dir):
    """
    Walk through product_images directory and collect all product folders.
    Returns dict mapping product_id to list of image paths.
    """
    product_dict = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_dir}' does not exist!")
        return product_dict
    
    # Walk through each folder in product_images
    for product_folder in base_path.iterdir():
        if not product_folder.is_dir():
            continue
        
        product_id = product_folder.name
        image_paths = []
        
        # Collect all supported images in this folder
        for img_file in product_folder.iterdir():
            if img_file.suffix in SUPPORTED_EXTENSIONS:
                image_paths.append(img_file)
        
        if image_paths:
            product_dict[product_id] = sorted(image_paths)
            print(f"Found {len(image_paths)} images for product '{product_id}'")
    
    return product_dict


def main():
    """Main function to precompute ORB features."""
    print("=" * 60)
    print("ORB Feature Precomputation")
    print("=" * 60)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    print(f"Initialized ORB with nfeatures={ORB_NFEATURES}")
    
    # Walk through product images
    print(f"\nScanning '{PRODUCT_IMAGES_DIR}' directory...")
    product_dict = walk_product_images(PRODUCT_IMAGES_DIR)
    
    if not product_dict:
        print(f"\nError: No product folders found in '{PRODUCT_IMAGES_DIR}'!")
        print("Please ensure the directory exists and contains product folders.")
        return
    
    # Process each product
    refs_data = {}
    total_images = 0
    total_successful = 0
    
    print(f"\nProcessing {len(product_dict)} products...")
    print("-" * 60)
    
    for product_id, image_paths in product_dict.items():
        refs_data[product_id] = []
        
        for img_path in image_paths:
            total_images += 1
            print(f"Processing: {img_path}...", end=" ")
            
            # Load image
            image = load_image(img_path)
            if image is None:
                print("FAILED (could not load image)")
                continue
            
            # Compute ORB features
            keypoints, descriptors = compute_orb_features(image, orb)
            
            if descriptors is None or len(keypoints) == 0:
                print("FAILED (no features detected)")
                continue
            
            # Store reference data
            # Note: keypoints cannot be pickled directly, so we convert to a serializable format
            kp_data = [(kp.pt[0], kp.pt[1], kp.angle, kp.response, kp.size, kp.octave) 
                      for kp in keypoints]
            
            refs_data[product_id].append({
                "path": str(img_path),
                "kp": kp_data,
                "des": descriptors
            })
            
            total_successful += 1
            print(f"OK ({len(keypoints)} features)")
    
    # Save to pickle file
    print("\n" + "-" * 60)
    print(f"Saving to '{OUTPUT_FILE}'...")
    
    try:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(refs_data, f)
        print(f"Successfully saved reference data to '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Products processed: {len(refs_data)}")
    print(f"  Total images: {total_images}")
    print(f"  Successfully processed: {total_successful}")
    print(f"  Failed: {total_images - total_successful}")
    
    # Print per-product summary
    print("\nPer-product breakdown:")
    for product_id, refs in refs_data.items():
        total_features = sum(len(ref["kp"]) for ref in refs)
        print(f"  {product_id}: {len(refs)} images, {total_features} total features")
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()

