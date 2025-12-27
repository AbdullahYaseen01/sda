"""
test_detection_cli.py - CLI test script for ResNet50 product detection

This script allows you to test product detection on individual images without
needing to run the full webcam system.

Usage:
    python test_detection_cli.py

Then paste image paths when prompted.
"""

import os
import cv2
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFS_FILE = "refs_resnet.pkl"
PRODUCTS_CSV = "products.csv"
MIN_SIMILARITY_SCORE = 0.80  # Increased from 0.65 to reduce false positives

class ProductDetectorCLI:
    """Product detector for CLI testing."""
    
    def __init__(self):
        """Initialize detector."""
        print("Initializing detector...")
        self.device = DEVICE
        self.refs_data = None
        self.products_df = None
        
        # Load feature extractor
        self.feature_extractor = self._load_feature_extractor()
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        self.load_references()
        self.load_products()
        print("✓ Detector ready!\n")
    
    def _load_feature_extractor(self) -> nn.Module:
        """Load pre-trained ResNet50."""
        print("Loading pre-trained ResNet50...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.to(self.device)
        feature_extractor.eval()
        return feature_extractor
    
    def load_references(self):
        """Load precomputed features."""
        if not os.path.exists(REFS_FILE):
            print(f"ERROR: '{REFS_FILE}' not found!")
            print(f"Run: python precompute_refs_resnet.py")
            raise FileNotFoundError(REFS_FILE)
        
        with open(REFS_FILE, 'rb') as f:
            self.refs_data = pickle.load(f)
        
        print(f"Loaded references for {len(self.refs_data)} products")
    
    def load_products(self):
        """Load product CSV."""
        if not os.path.exists(PRODUCTS_CSV):
            print(f"WARNING: '{PRODUCTS_CSV}' not found (optional)")
            self.products_df = pd.DataFrame()
            return
        
        try:
            self.products_df = pd.read_csv(PRODUCTS_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            self.products_df = pd.read_csv(PRODUCTS_CSV, encoding='latin-1')
        
        print(f"Loaded {len(self.products_df)} products from CSV")
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
            
            features = features.squeeze().cpu().numpy()
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(features1, features2))
    
    def detect_product(self, image_path: str, single_product_mode: bool = True) -> List[Dict]:
        """Detect products in image.
        
        Args:
            image_path: Path to image file
            single_product_mode: If True, match whole image. If False, use region proposals.
        """
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image")
            return []
        
        height, width = image.shape[:2]
        
        if single_product_mode:
            # Single product: extract features from whole image
            features = self.extract_features(image_path)
            if features is None:
                return []
            
            # Find best match
            best_match = None
            best_score = 0
            
            for product_name, product_images in self.refs_data.items():
                for ref_image_data in product_images:
                    ref_features = ref_image_data['features']
                    similarity = self.compute_similarity(features, ref_features)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = product_name
            
            if best_match:
                return [{'product_name': best_match, 'score': best_score}]
            return []
        
        else:
            # Multi-product: use region proposals
            return self._detect_regions(image)
    
    def _detect_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect products using region proposals (optimized for speed)."""
        height, width = image.shape[:2]
        detections = []
        
        # Optimized parameters for speed
        scales = [0.5, 0.65, 0.8]  # Reduced from 6 to 3 scales
        stride = 80  # Increased from 40 to 80 (skip more regions)
        min_box_size = 40
        threshold = 0.70
        max_proposals = 50  # Stop after 50 proposals
        
        print("Scanning regions...")
        region_count = 0
        
        for scale in scales:
            box_size = int(224 * scale)
            
            if box_size < min_box_size:
                continue
            
            # Sliding window with larger stride
            for y in range(0, height - box_size + 1, stride):
                for x in range(0, width - box_size + 1, stride):
                    if region_count >= max_proposals:
                        break
                    
                    region_count += 1
                    
                    x1, y1 = x, y
                    x2, y2 = min(x + box_size, width), min(y + box_size, height)
                    
                    # Extract region
                    region = image[y1:y2, x1:x2]
                    
                    # Extract features
                    region_features = self._extract_region_features(region)
                    if region_features is None:
                        continue
                    
                    # Find best matching product
                    best_match = None
                    best_score = threshold
                    
                    for product_name, product_images in self.refs_data.items():
                        for ref_image_data in product_images:
                            ref_features = ref_image_data['features']
                            similarity = self.compute_similarity(region_features, ref_features)
                            
                            if similarity > best_score:
                                best_score = similarity
                                best_match = product_name
                    
                    if best_match is not None:
                        detections.append({
                            'product_name': best_match,
                            'box': (x1, y1, x2, y2),
                            'score': best_score
                        })
                    
                    # Print progress
                    if region_count % 10 == 0:
                        print(f"  Scanned {region_count} regions, found {len(detections)} matches...")
                
                if region_count >= max_proposals:
                    break
        
        print(f"Scanned {region_count} regions, found {len(detections)} matches")
        
        # Apply NMS
        detections = self._nms(detections)
        
        # Convert to result format
        results = []
        for det in detections:
            results.append({
                'product_name': det['product_name'],
                'score': det['score']
            })
        
        return results
    
    def _extract_region_features(self, region: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a numpy array region."""
        try:
            # Convert BGR to RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(region_rgb)
            
            # Apply transforms
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
            
            # Convert to numpy and flatten
            features = features.squeeze().cpu().numpy()
            
            # Normalize to unit length
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except:
            return None
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return detections
        
        # Sort by score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        nms_threshold = 0.4
        
        for det in detections:
            keep_this = True
            x1_i, y1_i, x2_i, y2_i = det['box']
            
            for kept_det in keep:
                x1_j, y1_j, x2_j, y2_j = kept_det['box']
                
                # Compute IoU
                inter_x1 = max(x1_i, x1_j)
                inter_y1 = max(y1_i, y1_j)
                inter_x2 = min(x2_i, x2_j)
                inter_y2 = min(y2_i, y2_j)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_i = (x2_i - x1_i) * (y2_i - y1_i)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    union_area = area_i + area_j - inter_area
                    iou = inter_area / union_area
                    
                    if iou > nms_threshold:
                        keep_this = False
                        break
            
            if keep_this:
                keep.append(det)
        
        return keep
    
    def get_product_info(self, product_name: str) -> Optional[Dict]:
        """Get product info from CSV."""
        if self.products_df.empty:
            return None
        
        matches = self.products_df[self.products_df['name'] == product_name]
        return matches.iloc[0].to_dict() if not matches.empty else None


def main():
    """Main CLI loop."""
    print("=" * 70)
    print("ResNet50 Product Detection - CLI Test (Single Product Mode)")
    print("=" * 70)
    
    try:
        detector = ProductDetectorCLI()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return
    
    print("Commands:")
    print("  - Paste image path and press Enter (single product mode)")
    print("  - Type 'm' to switch to multi-product mode")
    print("  - Type 'quit' or 'q' to exit")
    print("=" * 70)
    
    # Mode: single or multi product detection
    single_product_mode = True
    
    while True:
        try:
            image_path = input("\nEnter image path (or 'q' to quit): ").strip()
            
            if not image_path:
                continue
            
            if image_path.lower() in ['q', 'quit']:
                print("Goodbye!")
                break
            
            if image_path.lower() == 'm':
                single_product_mode = not single_product_mode
                mode_name = "single product" if single_product_mode else "multi-product"
                print(f"\nSwitched to {mode_name} mode")
                continue
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"ERROR: File not found: {image_path}")
                continue
            
            # Detect
            results = detector.detect_product(image_path, single_product_mode=single_product_mode)
            
            if results:
                num_products = len(results)
                print(f"\n✓ Detected {num_products} product(s):\n")
                
                for i, result in enumerate(results, 1):
                    name = result['product_name']
                    score = result['score']
                    
                    # Get price if available
                    product_info = detector.get_product_info(name)
                    price = ""
                    if product_info:
                        price_val = product_info.get('price', product_info.get('price_excl_vat', 'N/A'))
                        price = f" | Price: ${price_val}"
                    
                    print(f"  {i}. {name}")
                    print(f"     Score: {score:.4f} (Confidence: {score*100:.1f}%){price}\n")
            else:
                print("\n✗ No products detected")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
