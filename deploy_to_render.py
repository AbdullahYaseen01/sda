"""
Automated deployment script for Render using Render API.
This script will deploy your application to Render automatically.

Prerequisites:
1. Render API key (get from https://dashboard.render.com/account/api-keys)
2. Python requests library: pip install requests
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
RENDER_API_KEY = ""  # You need to set this - get from Render dashboard
RENDER_API_BASE = "https://api.render.com/v1"
REPO_URL = "https://github.com/AbdullahYaseen01/sda"
SERVICE_NAME = "checkout-mvp"
REGION = "frankfurt"  # Options: frankfurt, oregon, singapore

def get_headers():
    """Get API headers with authentication."""
    if not RENDER_API_KEY:
        print("[ERROR] RENDER_API_KEY is not set!")
        print("[INFO] Get your API key from: https://dashboard.render.com/account/api-keys")
        print("[INFO] Then edit this script and set RENDER_API_KEY = 'your-key-here'")
        sys.exit(1)
    
    return {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

def check_api_key():
    """Check if API key is valid."""
    print("[CHECK] Checking API key...")
    headers = get_headers()
    response = requests.get(f"{RENDER_API_BASE}/owners", headers=headers)
    
    if response.status_code == 200:
        print("[OK] API key is valid!")
        return True
    else:
        print(f"[ERROR] API key check failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def get_owner_id():
    """Get the owner (user/team) ID."""
    print("[INFO] Getting owner ID...")
    headers = get_headers()
    response = requests.get(f"{RENDER_API_BASE}/owners", headers=headers)
    
    if response.status_code == 200:
        owners = response.json()
        if owners:
            owner_id = owners[0]['owner']['id']
            print(f"[OK] Found owner ID: {owner_id}")
            return owner_id
        else:
            print("[ERROR] No owners found")
            return None
    else:
        print(f"[ERROR] Failed to get owner: {response.status_code}")
        return None

def create_service(owner_id):
    """Create a new web service on Render."""
    print(f"[DEPLOY] Creating service '{SERVICE_NAME}'...")
    
    headers = get_headers()
    
    service_config = {
        "type": "web_service",
        "name": SERVICE_NAME,
        "ownerId": owner_id,
        "repo": REPO_URL,
        "branch": "main",
        "serviceDetails": {
            "runtime": "python",
            "region": REGION,
            "envSpecificDetails": {
                "buildCommand": "pip install -r requirements_resnet.txt",
                "startCommand": "python app.py",
                "envVars": [
                    {
                        "key": "PYTHON_VERSION",
                        "value": "3.11.0"
                    }
                ]
            }
        }
    }
    
    response = requests.post(
        f"{RENDER_API_BASE}/services",
        headers=headers,
        json=service_config
    )
    
    if response.status_code == 201:
        service = response.json()
        service_id = service['service']['id']
        print(f"[OK] Service created! ID: {service_id}")
        url = service['service'].get('serviceDetails', {}).get('url', 'N/A')
        if url != 'N/A':
            print(f"[URL] Service URL: {url}")
        return service_id
    else:
        print(f"[ERROR] Failed to create service: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def check_service_status(service_id):
    """Check the status of a service."""
    headers = get_headers()
    response = requests.get(f"{RENDER_API_BASE}/services/{service_id}", headers=headers)
    
    if response.status_code == 200:
        service = response.json()
        return service['service'].get('serviceDetails', {}).get('deployStatus', 'unknown')
    return "unknown"

def wait_for_deployment(service_id, timeout=1800):
    """Wait for deployment to complete."""
    print("[WAIT] Waiting for deployment to complete...")
    print("       (This may take 10-15 minutes for first build)")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        status = check_service_status(service_id)
        
        if status != last_status:
            print(f"[STATUS] {status}")
            last_status = status
        
        if status == "live":
            print("[OK] Deployment complete! Service is live!")
            return True
        elif status == "build_failed":
            print("[ERROR] Build failed. Check logs in Render dashboard.")
            return False
        
        time.sleep(30)  # Check every 30 seconds
    
    print("[TIMEOUT] Timeout waiting for deployment")
    return False

def get_service_url(service_id):
    """Get the service URL."""
    headers = get_headers()
    response = requests.get(f"{RENDER_API_BASE}/services/{service_id}", headers=headers)
    
    if response.status_code == 200:
        service = response.json()
        url = service['service'].get('serviceDetails', {}).get('url', None)
        return url
    return None

def main():
    """Main deployment function."""
    print("=" * 60)
    print("Automated Render Deployment")
    print("=" * 60)
    print()
    
    # Check if API key is set
    if not RENDER_API_KEY:
        print("[ERROR] RENDER_API_KEY is not set!")
        print()
        print("[INFO] To get your API key:")
        print("   1. Go to: https://dashboard.render.com/account/api-keys")
        print("   2. Click 'Create API Key'")
        print("   3. Copy the key")
        print("   4. Edit this script and set: RENDER_API_KEY = 'your-key-here'")
        print()
        print("[INFO] Or set it as environment variable:")
        print("   export RENDER_API_KEY='your-key-here'")
        print("   python deploy_to_render.py")
        sys.exit(1)
    
    # Check API key validity
    if not check_api_key():
        sys.exit(1)
    
    # Get owner ID
    owner_id = get_owner_id()
    if not owner_id:
        sys.exit(1)
    
    # Create service
    service_id = create_service(owner_id)
    if not service_id:
        sys.exit(1)
    
    # Wait for deployment
    if wait_for_deployment(service_id):
        url = get_service_url(service_id)
        if url:
            print()
            print("=" * 60)
            print("DEPLOYMENT SUCCESSFUL!")
            print("=" * 60)
            print(f"[URL] Your app is live at: {url}")
            print(f"[HEALTH] Health check: {url}/api/health")
            print()
            print("[OK] Deployment complete!")
        else:
            print("[WARN] Service deployed but URL not available yet")
            print("       Check Render dashboard for the URL")
    else:
        print()
        print("[ERROR] Deployment did not complete successfully")
        print("   Check Render dashboard for details")

if __name__ == "__main__":
    # Check for API key in environment variable
    import os
    if os.environ.get("RENDER_API_KEY"):
        RENDER_API_KEY = os.environ.get("RENDER_API_KEY")
    
    main()

