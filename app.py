"""
Flask web application for product detection checkout system.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from datetime import datetime
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import List, Dict, Optional

app = Flask(__name__, template_folder='templates')
CORS(app)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFS_FILE = "refs_resnet.pkl"
PRODUCTS_CSV = "products.csv"
RECEIPT_DIR = "receipt"
MIN_SIMILARITY_SCORE = 0.40  # Lowered threshold for better detection
VAT_RATE = 0.20

# Global detector instance
detector = None
cart = None


class ProductDetector:
    """Product detector using ResNet50 features."""
    
    def __init__(self):
        """Initialize detector."""
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
    
    def _load_feature_extractor(self) -> nn.Module:
        """Load pre-trained ResNet50."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.to(self.device)
        feature_extractor.eval()
        return feature_extractor
    
    def load_references(self):
        """Load precomputed features."""
        if not Path(REFS_FILE).exists():
            raise FileNotFoundError(f"'{REFS_FILE}' not found! Run: python precompute_refs_resnet.py")
        
        with open(REFS_FILE, 'rb') as f:
            self.refs_data = pickle.load(f)
    
    def load_products(self):
        """Load product CSV."""
        if not Path(PRODUCTS_CSV).exists():
            self.products_df = pd.DataFrame()
            return
        
        try:
            self.products_df = pd.read_csv(PRODUCTS_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            self.products_df = pd.read_csv(PRODUCTS_CSV, encoding='latin-1')
    
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
            # Return best match even if below threshold (frontend can decide)
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
    
    def remove_item(self, product_name: str):
        """Remove item from cart."""
        self.items = [item for item in self.items if item['name'] != product_name]
    
    def clear(self):
        """Clear cart."""
        self.items = []
    
    def get_total(self) -> float:
        """Calculate total price."""
        total = 0
        for item in self.items:
            price = item.get('price', item.get('price_excl_vat', 0))
            total += float(price) * item['quantity']
        return total
    
    def get_total_with_vat(self) -> float:
        """Calculate total with VAT."""
        total = self.get_total()
        return total * (1 + VAT_RATE)
    
    def get_vat(self) -> float:
        """Calculate VAT amount."""
        return self.get_total_with_vat() - self.get_total()
    
    def to_dict(self) -> Dict:
        """Convert cart to dictionary."""
        return {
            'items': self.items,
            'subtotal': self.get_total(),
            'vat': self.get_vat(),
            'total': self.get_total_with_vat(),
            'item_count': len(self.items)
        }


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
            body {{ font-family: Arial, sans-serif; max-width: 500px; margin: 20px auto; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
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


def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
    """Decode base64 image string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image")
            return None
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def scan_product():
    """Scan product from image."""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        frame = decode_base64_image(image_data)
        
        if frame is None or frame.size == 0:
            return jsonify({
                'success': False,
                'message': 'Failed to decode image'
            }), 400
        
        # Detect product
        results = detector.detect_product(frame)
        
        if not results:
            return jsonify({
                'success': False,
                'message': 'No product detected'
            })
        
        result = results[0]
        product_name = result['product_name']
        score = result['score']
        
        # Get product info
        product_info = detector.get_product_info(product_name)
        
        if not product_info:
            return jsonify({
                'success': False,
                'message': 'Product info not found'
            })
        
        # Check if confidence is high enough
        if score < MIN_SIMILARITY_SCORE:
            return jsonify({
                'success': False,
                'message': f'Low confidence: {score:.2f} (minimum: {MIN_SIMILARITY_SCORE})',
                'detected_product': product_name,
                'confidence': score
            })
        
        # Add to cart
        cart.add_item(product_info)
        
        return jsonify({
            'success': True,
            'product': {
                'name': product_name,
                'price': product_info.get('price', product_info.get('price_excl_vat', 0)),
                'category': product_info.get('category', ''),
                'confidence': score
            },
            'cart': cart.to_dict()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cart', methods=['GET'])
def get_cart():
    """Get current cart."""
    return jsonify(cart.to_dict())


@app.route('/api/cart/clear', methods=['POST'])
def clear_cart():
    """Clear cart."""
    cart.clear()
    return jsonify({'success': True, 'cart': cart.to_dict()})


@app.route('/api/cart/remove', methods=['POST'])
def remove_item():
    """Remove item from cart."""
    data = request.json
    product_name = data.get('product_name')
    
    if not product_name:
        return jsonify({'error': 'Product name required'}), 400
    
    cart.remove_item(product_name)
    return jsonify({'success': True, 'cart': cart.to_dict()})


@app.route('/api/receipt', methods=['POST'])
def generate_receipt():
    """Generate receipt."""
    try:
        html_content = generate_receipt_html(cart)
        receipt_path = save_receipt(html_content)
        
        return jsonify({
            'success': True,
            'receipt_path': receipt_path,
            'receipt_html': html_content
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        'status': 'ok',
        'detector_ready': detector is not None,
        'products_loaded': len(detector.refs_data) if detector else 0
    })


if __name__ == '__main__':
    # Initialize detector and cart
    print("Initializing product detector...")
    detector = ProductDetector()
    cart = Cart()
    print("Detector ready!")
    
    # Run Flask app
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

