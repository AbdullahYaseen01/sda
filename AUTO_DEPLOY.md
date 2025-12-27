# 🤖 Automated Deployment to Render

I've created an automated deployment script that will deploy your app to Render with minimal effort!

## ⚡ Quick Start (3 Steps)

### Step 1: Get Render API Key

1. Go to: **https://dashboard.render.com/account/api-keys**
2. Click **"Create API Key"**
3. Give it a name (e.g., "Deployment Key")
4. **Copy the key** (you'll only see it once!)

### Step 2: Set API Key

**Option A: Edit the script**
```bash
# Open deploy_to_render.py
# Find: RENDER_API_KEY = ""
# Change to: RENDER_API_KEY = "your-key-here"
```

**Option B: Use environment variable**
```bash
# Windows PowerShell:
$env:RENDER_API_KEY="your-key-here"

# Windows CMD:
set RENDER_API_KEY=your-key-here

# Linux/Mac:
export RENDER_API_KEY="your-key-here"
```

### Step 3: Run the Script

```bash
# Install requests if needed
pip install requests

# Run deployment
python deploy_to_render.py
```

**That's it!** The script will:
- ✅ Check your API key
- ✅ Create the service
- ✅ Configure everything
- ✅ Deploy your app
- ✅ Wait for completion
- ✅ Give you the URL

---

## 📋 What the Script Does

1. **Validates API Key** - Checks if your key works
2. **Gets Owner ID** - Finds your Render account
3. **Creates Service** - Sets up the web service with correct config
4. **Deploys** - Starts the deployment
5. **Monitors** - Waits for build to complete (10-15 min)
6. **Reports** - Gives you the live URL

---

## 🔧 Configuration

The script is pre-configured with:
- **Repository**: `https://github.com/AbdullahYaseen01/sda`
- **Service Name**: `checkout-mvp`
- **Region**: `frankfurt` (change if needed)
- **Build Command**: `pip install -r requirements_resnet.txt`
- **Start Command**: `python app.py`
- **Plan**: `starter` (change to `free` for free tier)

To change settings, edit `deploy_to_render.py`.

---

## ⚠️ Important Notes

### Free Tier
If you want to use the free tier, change this line in the script:
```python
"planId": "free",  # Change from "starter" to "free"
```

### Region
Change region if needed:
```python
REGION = "oregon"  # Options: frankfurt, oregon, singapore
```

---

## 🐛 Troubleshooting

### "API key is not set"
- Make sure you set `RENDER_API_KEY` in the script or environment

### "API key check failed"
- Check that your API key is correct
- Make sure you copied the full key
- Try creating a new API key

### "Failed to create service"
- Check that the repository is public or you've authorized Render
- Verify the repository URL is correct
- Check Render dashboard for error details

### Build takes too long
- First build takes 10-15 minutes (downloading PyTorch)
- This is normal! Just wait.

---

## 📊 Script Output

You'll see:
```
🔑 Checking API key...
✅ API key is valid!
👤 Getting owner ID...
✅ Found owner ID: abc123
🚀 Creating service 'checkout-mvp'...
✅ Service created! ID: srv-xyz789
⏳ Waiting for deployment to complete...
📊 Status: building
📊 Status: live
✅ Deployment complete! Service is live!

🎉 DEPLOYMENT SUCCESSFUL!
🌐 Your app is live at: https://checkout-mvp.onrender.com
```

---

## 🎯 Alternative: Manual Deployment

If the script doesn't work, use the manual guide:
- See `RENDER_DEPLOY_NOW.md` for step-by-step instructions

---

## ✅ After Deployment

1. **Test your app**: Visit the URL provided
2. **Check health**: `https://your-app.onrender.com/api/health`
3. **Test scanning**: Try the product detection feature

---

## 🚀 Ready to Deploy?

1. Get API key: https://dashboard.render.com/account/api-keys
2. Set it in the script or environment
3. Run: `python deploy_to_render.py`
4. Wait 10-15 minutes
5. Done! 🎉

---

**The script does everything automatically - just get the API key and run it!**

