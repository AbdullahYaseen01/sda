"""
scan_and_detect_hybrid.py - Webcam + CLI-based product detection

This script shows live webcam feed, captures snapshots on ENTER, and detects
products using the accurate CLI detection method. Combines reliability of CLI
with convenience of webcam interface.

Usage:
    python scan_and_detect_hybrid.py

Controls:
    - Press ENTER: Capture snapshot and detect products
    - Press 'c': Continue (skip frame)
    - Press 'q': Quit and generate receipt
"""

import cv2
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFS_FILE = "refs_resnet.pkl"
PRODUCTS_CSV = "products.csv"
RECEIPT_DIR = "receipt"
MIN_SIMILARITY_SCORE = 0.80
VAT_RATE = 0.20


class ProductDetector:
    """Product detector using ResNet50 features."""
    
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
        print("[OK] Detector ready!\n")
    
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
        if not Path(REFS_FILE).exists():
            print(f"ERROR: '{REFS_FILE}' not found!")
            print(f"Run: python precompute_refs_resnet.py")
            raise FileNotFoundError(REFS_FILE)
        
        with open(REFS_FILE, 'rb') as f:
            self.refs_data = pickle.load(f)
        
        print(f"Loaded references for {len(self.refs_data)} products")
    
    def load_products(self):
        """Load product CSV."""
        if not Path(PRODUCTS_CSV).exists():
            print(f"WARNING: '{PRODUCTS_CSV}' not found")
            self.products_df = pd.DataFrame()
            return
        
        try:
            self.products_df = pd.read_csv(PRODUCTS_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            self.products_df = pd.read_csv(PRODUCTS_CSV, encoding='latin-1')
        
        print(f"Loaded {len(self.products_df)} products from CSV")
    
    def extract_features(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract ResNet50 features from image array."""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
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
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(features1, features2))
    
    def detect_product(self, frame: np.ndarray) -> List[Dict]:
        """Detect product from frame snapshot."""
        # Extract features from frame
        features = self.extract_features(frame)
        if features is None:
            return []
        
        # Find best matches
        matches = []
        
        for product_name, product_images in self.refs_data.items():
            best_score = 0
            
            for ref_image_data in product_images:
                ref_features = ref_image_data['features']
                similarity = self.compute_similarity(features, ref_features)
                best_score = max(best_score, similarity)
            
            matches.append({
                'product_name': product_name,
                'score': best_score
            })
        
        # Sort by score and return top match
        matches = sorted(matches, key=lambda x: x['score'], reverse=True)
        
        if matches:
            return [matches[0]]  # Return best match
        return []
    
    def get_product_info(self, product_name: str) -> Optional[Dict]:
        """Get product info from CSV."""
        if self.products_df.empty:
            return None
        
        matches = self.products_df[self.products_df['name'] == product_name]
        return matches.iloc[0].to_dict() if not matches.empty else None


class Cart:
    """Shopping cart."""
    
    def __init__(self):
        self.items = []
    
    def add_item(self, product_info: Dict):
        """Add item to cart."""
        for item in self.items:
            if item['name'] == product_info['name']:
                item['quantity'] += 1
                return
        
        product_info['quantity'] = 1
        self.items.append(product_info)
    
    def get_total(self) -> float:
        """Calculate total price."""
        total = 0
        for item in self.items:
            price = item.get('price', item.get('price_excl_vat', 0))
            total += float(price) * item['quantity']
        return total
    
    def get_total_with_vat(self) -> float:
        """Calculate total with VAT."""
        return self.get_total() * (1 + VAT_RATE)
    
    def get_vat(self) -> float:
        """Calculate VAT amount."""
        return self.get_total_with_vat() - self.get_total()


def generate_receipt_html(cart: Cart) -> str:
    """Generate HTML receipt."""
    timestamp = datetime.now()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Receipt</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 500px; margin: 20px auto; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .total {{ font-weight: bold; font-size: 16px; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Receipt</h2>
            <p>{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <table>
            <tr>
                <th>Product</th>
                <th>Qty</th>
                <th>Unit Price</th>
                <th>Total</th>
            </tr>
    """
    
    for item in cart.items:
        price = item.get('price', item.get('price_excl_vat', 0))
        total = float(price) * item['quantity']
        html += f"""
            <tr>
                <td>{item.get('name', 'Unknown')}</td>
                <td>{item['quantity']}</td>
                <td>${price:.2f}</td>
                <td>${total:.2f}</td>
            </tr>
        """
    
    html += f"""
        </table>
        <table style="margin-top: 20px;">
            <tr>
                <td>Subtotal:</td>
                <td class="total">${cart.get_total():.2f}</td>
            </tr>
            <tr>
                <td>VAT ({VAT_RATE*100:.0f}%):</td>
                <td class="total">${cart.get_vat():.2f}</td>
            </tr>
            <tr style="font-size: 18px;">
                <td>Total:</td>
                <td class="total">${cart.get_total_with_vat():.2f}</td>
            </tr>
        </table>
        <div class="footer">
            <p>Thank you for your purchase!</p>
        </div>
    </body>
    </html>
    """
    
    return html


def save_receipt(html_content: str) -> str:
    """Save receipt to file."""
    Path(RECEIPT_DIR).mkdir(exist_ok=True)
    
    timestamp = datetime.now()
    filename = f"receipt_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    filepath = Path(RECEIPT_DIR) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(filepath)


def main():
    """Main function."""
    print("=" * 70)
    print("Product Detection - Webcam + Snapshot Mode")
    print("=" * 70)
    
    try:
        detector = ProductDetector()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    cart = Cart()
    
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Increase resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "=" * 70)
    print("Controls:")
    print("  - Press ENTER: Capture & detect product")
    print("  - Press 'c': Continue (skip frame)")
    print("  - Press 'q': Quit and generate receipt")
    print("=" * 70 + "\n")
    
    frame_count = 0
    last_detection = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Display frame
        display_frame = frame.copy()
        
        # Show last detection on frame
        if last_detection:
            product_name = last_detection['product_name']
            score = last_detection['score']
            text = f"{product_name} ({score:.2f})"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        # Show cart count
        cv2.putText(display_frame, f"Cart: {len(cart.items)} items", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Product Scanner - Press ENTER to scan', display_frame)
        
        # Wait for key
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            print("\n[SCANNING]")
            
            # Detect
            results = detector.detect_product(frame)
            
            if results:
                result = results[0]
                product_name = result['product_name']
                score = result['score']
                
                print(f"[OK] Detected: {product_name}")
                print(f"  Confidence: {score:.4f} ({score*100:.1f}%)")
                
                # Get product info and add to cart
                product_info = detector.get_product_info(product_name)
                if product_info:
                    price = product_info.get('price', product_info.get('price_excl_vat', 0))
                    print(f"  Price: ${price}")
                    cart.add_item(product_info)
                    print(f"  Cart total: {len(cart.items)} items - ${cart.get_total_with_vat():.2f}")
                    last_detection = result
                else:
                    print(f"  WARNING: No price info found!")
            else:
                print("✗ No product detected")
                last_detection = None
        
        elif key == ord('c'):
            continue
        
        elif key == ord('q'):
            print("\n[GENERATING RECEIPT]")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate receipt
    if cart.items:
        print(f"\nReceipt Summary:")
        print(f"Items: {len(cart.items)}")
        print(f"Subtotal: ${cart.get_total():.2f}")
        print(f"VAT: ${cart.get_vat():.2f}")
        print(f"Total: ${cart.get_total_with_vat():.2f}")
        
        html_content = generate_receipt_html(cart)
        receipt_path = save_receipt(html_content)
        
        print(f"\nReceipt saved to: {receipt_path}")
    else:
        print("Cart is empty. No receipt generated.")


if __name__ == "__main__":
    main()
