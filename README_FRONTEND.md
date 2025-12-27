# Product Checkout System - Web Frontend

A modern web-based frontend for the product detection checkout system.

## Features

- 🎥 **Live Webcam Feed** - Real-time product scanning using your device's camera
- 🛒 **Shopping Cart** - Add, remove, and manage scanned products
- 💰 **Automatic Pricing** - Calculates subtotal, VAT, and total automatically
- 🧾 **Receipt Generation** - Generate and view HTML receipts
- 📱 **Responsive Design** - Works on desktop and mobile devices
- 🎨 **Modern UI** - Beautiful gradient design with smooth animations

## Installation

1. Install dependencies:
```bash
pip install -r requirements_resnet.txt
```

2. Make sure you have precomputed the ResNet features:
```bash
python precompute_refs_resnet.py
```

## Running the Application

Start the Flask server:
```bash
python app.py
```

The application will be available at:
- **Frontend**: http://localhost:5000
- **API**: http://localhost:5000/api/

## Usage

1. **Open the application** in your web browser (Chrome, Firefox, Edge recommended)
2. **Allow camera access** when prompted
3. **Position products** in front of the camera
4. **Click "Scan Product"** to detect and add to cart
5. **View your cart** on the right side panel
6. **Remove items** by clicking the × button
7. **Generate receipt** by clicking "Generate Receipt" or "Checkout"

## API Endpoints

- `GET /` - Main frontend page
- `POST /api/scan` - Scan product from image
- `GET /api/cart` - Get current cart
- `POST /api/cart/clear` - Clear cart
- `POST /api/cart/remove` - Remove item from cart
- `POST /api/receipt` - Generate receipt
- `GET /api/health` - Health check

## Requirements

- Python 3.8+
- Webcam/Camera access
- Modern web browser with WebRTC support
- Precomputed ResNet features (`refs_resnet.pkl`)
- Products CSV file (`products.csv`)

## Troubleshooting

**Camera not working:**
- Make sure you've allowed camera access in your browser
- Try using HTTPS (some browsers require it for camera access)
- Check if another application is using the camera

**No products detected:**
- Ensure products are well-lit and clearly visible
- Make sure `refs_resnet.pkl` exists and is up to date
- Check that products in the image match products in your database

**API errors:**
- Verify that `products.csv` exists
- Check that `refs_resnet.pkl` was generated successfully
- Review the console for error messages


