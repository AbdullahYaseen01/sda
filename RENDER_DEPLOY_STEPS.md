# Deploy to Render - Step-by-Step Guide

## 🚀 Quick Deploy (5 Minutes)

### Step 1: Create Render Account

1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with your **GitHub account** (recommended)
   - This allows automatic deployment from your GitHub repo

### Step 2: Create New Web Service

1. In Render dashboard, click **"New +"** button (top right)
2. Select **"Web Service"**
3. You'll see two options:
   - **"Public Git repository"** - Enter your repo URL
   - **"Connect account"** - Connect your GitHub account

### Step 3: Connect Your Repository

**Option A: Connect GitHub Account (Recommended)**
1. Click **"Connect account"**
2. Authorize Render to access your GitHub
3. Select repository: **`AbdullahYaseen01/sda`**
4. Click **"Connect"**

**Option B: Use Public Git Repository**
1. Select **"Public Git repository"**
2. Enter: `https://github.com/AbdullahYaseen01/sda`
3. Click **"Continue"**

### Step 4: Configure Service

Fill in the following settings:

- **Name**: `checkout-mvp` (or any name you prefer)
- **Region**: Choose closest to you (e.g., `Frankfurt`, `Oregon`, `Singapore`)
- **Branch**: `main`
- **Root Directory**: (leave empty)
- **Runtime**: `Python 3`
- **Build Command**: 
  ```
  pip install -r requirements_resnet.txt
  ```
- **Start Command**: 
  ```
  python app.py
  ```
- **Plan**: 
  - **Free** - Spins down after 15 min inactivity (first request slow)
  - **Starter ($7/month)** - Always on, better performance

### Step 5: Environment Variables (Optional)

Click **"Advanced"** → **"Environment Variables"**:

- `PYTHON_VERSION` = `3.11.0` (optional, Render auto-detects)
- `PORT` = `5000` (optional, app.py reads from environment)

**Note**: The app already handles PORT automatically, so you can skip this.

### Step 6: Deploy

1. Click **"Create Web Service"** at the bottom
2. Render will start building your app
3. **First build takes 10-15 minutes** (downloading PyTorch)
4. Watch the build logs in real-time

### Step 7: Wait for Deployment

- Build logs will show progress
- You'll see: "Installing dependencies..."
- Then: "Starting service..."
- Finally: "Your service is live at: https://checkout-mvp.onrender.com"

### Step 8: Test Your App

1. Visit your app URL: `https://checkout-mvp.onrender.com`
2. Test health endpoint: `https://checkout-mvp.onrender.com/api/health`
3. Try scanning a product!

---

## ✅ What Happens During Deployment

1. **Clone Repository**: Render clones your GitHub repo
2. **Install Dependencies**: Runs `pip install -r requirements_resnet.txt`
   - Downloads PyTorch (~500MB) - takes 5-10 minutes
   - Installs Flask, OpenCV, etc.
3. **Start Application**: Runs `python app.py`
4. **Health Check**: Render checks if app is responding
5. **Go Live**: Your app is accessible via HTTPS URL

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- ✅ Code is pushed to GitHub (already done!)
- ✅ `requirements_resnet.txt` exists (already done!)
- ✅ `app.py` exists (already done!)
- ✅ `refs_resnet.pkl` is in repo (already done!)
- ✅ `products.csv` is in repo (already done!)
- ✅ `templates/index.html` exists (already done!)

**Everything is ready!** ✅

---

## 🐛 Troubleshooting

### Build Fails

**Issue**: Build fails during dependency installation
- **Solution**: Check build logs, ensure all dependencies are in `requirements_resnet.txt`

### App Crashes on Start

**Issue**: Service starts but crashes
- **Solution**: Check logs in Render dashboard
- Common issues:
  - Missing `refs_resnet.pkl` file
  - Missing `products.csv`
  - Port configuration issue

### Slow First Request (Free Tier)

**Issue**: First request takes 30-60 seconds
- **Solution**: This is normal for free tier (spins down after inactivity)
- **Fix**: Upgrade to Starter plan ($7/month) for always-on service

### Memory Errors

**Issue**: Out of memory errors
- **Solution**: Free tier has 512MB RAM (may not be enough for PyTorch)
- **Fix**: Upgrade to Starter plan (2GB RAM)

---

## 🔧 Advanced Configuration

### Custom Domain

1. Go to your service settings
2. Click **"Custom Domains"**
3. Add your domain
4. Follow DNS configuration instructions

### Environment Variables

If you need to change settings:

1. Go to service settings
2. Click **"Environment"**
3. Add variables:
   - `MIN_SIMILARITY_SCORE` = `0.40`
   - `VAT_RATE` = `0.20`

### Auto-Deploy

Render automatically deploys when you push to GitHub:
- Push to `main` branch → Auto-deploy
- Push to other branches → No deploy (unless configured)

---

## 📊 Monitoring

### View Logs

1. Go to your service dashboard
2. Click **"Logs"** tab
3. See real-time application logs

### Metrics

1. Go to service dashboard
2. Click **"Metrics"** tab
3. View:
   - CPU usage
   - Memory usage
   - Request count
   - Response times

---

## 💰 Pricing

### Free Tier
- ✅ 750 hours/month (enough for testing)
- ✅ Spins down after 15 min inactivity
- ⚠️ 512MB RAM (may be tight for PyTorch)
- ⚠️ Slow cold starts

### Starter Plan ($7/month)
- ✅ Always on
- ✅ 2GB RAM (better for PyTorch)
- ✅ Faster performance
- ✅ No cold starts

**Recommendation**: Start with Free tier, upgrade if needed.

---

## 🎯 Next Steps After Deployment

1. **Test your app**: Visit the URL
2. **Check health**: `/api/health` endpoint
3. **Test scanning**: Try scanning a product
4. **Monitor logs**: Watch for any errors
5. **Share URL**: Your app is live!

---

## 📞 Need Help?

If you encounter issues:

1. **Check Render logs**: Dashboard → Your Service → Logs
2. **Check build logs**: Dashboard → Your Service → Events
3. **Verify files**: Ensure all files are in GitHub repo
4. **Test locally**: Run `python app.py` locally first

---

## ✅ Summary

**Your app is ready to deploy!**

Just follow Steps 1-8 above, and your app will be live in 10-15 minutes!

**Repository**: https://github.com/AbdullahYaseen01/sda
**Render**: https://render.com

Good luck! 🚀

