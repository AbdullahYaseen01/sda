"""
Vercel serverless function wrapper for Flask app.
Note: This may not work well due to PyTorch size limitations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app

# Export for Vercel
def handler(request):
    return app(request.environ, lambda status, headers: None)

