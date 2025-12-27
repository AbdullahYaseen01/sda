# Deployment Guide - Product Checkout System

## ⚠️ Important: Netlify Limitation

**Netlify cannot host this application** because:
- Netlify is for static sites and serverless functions
- Your app requires a long-running Python server
- PyTorch dependencies are too large (~500MB+) for Netlify Functions
- Serverless functions have 10-second timeout (too short for ML inference)

## ✅ Recommended Platforms

### Option 1: Render (Recommended - Easiest)

**Pros:** Free tier, easy setup, supports Python apps with large dependencies

**Steps:**

1. **Create a Render account** at https://render.com

2. **Create a new Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Or upload your code directly

3. **Configure the service:**
   - **Name:** `checkout-mvp`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements_resnet.txt`
   - **Start Command:** `python app.py`
   - **Plan:** Free (or Starter for better performance)

4. **Add Environment Variables:**
   - `PORT` = `5000` (Render sets this automatically, but app.py reads it)

5. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy automatically
   - Your app will be available at: `https://checkout-mvp.onrender.com`

**Note:** Free tier spins down after 15 minutes of inactivity. First request may take 30-60 seconds to wake up.

---

### Option 2: Railway

**Pros:** Fast, good free tier, easy deployment

**Steps:**

1. **Create Railway account** at https://railway.app

2. **Create new project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo" (or upload code)

3. **Railway auto-detects Python:**
   - It will use `requirements_resnet.txt` automatically
   - Set start command: `python app.py`

4. **Deploy:**
   - Railway automatically deploys
   - Your app will be available at: `https://your-app-name.up.railway.app`

---

### Option 3: Fly.io

**Pros:** Global edge deployment, good performance

**Steps:**

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login:**
   ```bash
   fly auth login
   ```

3. **Create app:**
   ```bash
   fly launch
   ```

4. **Deploy:**
   ```bash
   fly deploy
   ```

---

### Option 4: Heroku

**Pros:** Well-established, good documentation

**Steps:**

1. **Install Heroku CLI**

2. **Login:**
   ```bash
   heroku login
   ```

3. **Create app:**
   ```bash
   heroku create checkout-mvp
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [x] `requirements_resnet.txt` includes all dependencies
- [x] `refs_resnet.pkl` is generated (run `python precompute_refs_resnet.py`)
- [x] `products.csv` exists
- [x] `app.py` reads `PORT` from environment (already fixed)
- [x] All necessary files are in the repository

## 🔧 Required Files for Deployment

Make sure these files are in your repository:

```
checkout_mvp/
├── app.py                    # Main Flask app
├── requirements_resnet.txt   # Dependencies
├── products.csv              # Product database
├── refs_resnet.pkl          # Precomputed features (IMPORTANT!)
├── products/                 # Product images directory
│   ├── Product 1/
│   └── Product 2/
├── templates/
│   └── index.html           # Frontend
└── Procfile                 # For Heroku/Railway (optional)
```

## ⚠️ Important Notes

1. **File Size Limits:**
   - `refs_resnet.pkl` can be large (50-200MB)
   - Some platforms have file size limits
   - Consider using Git LFS for large files

2. **Build Time:**
   - First build may take 10-15 minutes (downloading PyTorch)
   - Subsequent builds are faster (cached dependencies)

3. **Memory Requirements:**
   - PyTorch needs ~2-4GB RAM
   - Free tiers may have limitations
   - Consider upgrading if you get memory errors

4. **HTTPS Required:**
   - Webcam access requires HTTPS in most browsers
   - All recommended platforms provide HTTPS automatically

## 🚀 Quick Deploy to Render (Step-by-Step)

1. **Push code to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to Render Dashboard:**
   - Visit https://dashboard.render.com
   - Click "New +" → "Web Service"

3. **Connect Repository:**
   - Select your GitHub repository
   - Or use "Public Git repository" and paste your repo URL

4. **Configure:**
   - **Name:** `checkout-mvp`
   - **Region:** Choose closest to you
   - **Branch:** `main`
   - **Root Directory:** (leave empty)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements_resnet.txt`
   - **Start Command:** `python app.py`

5. **Advanced Settings:**
   - **Plan:** Free (or Starter for $7/month - no spin-down)
   - Add environment variable: `PORT=5000` (optional, app.py handles this)

6. **Deploy:**
   - Click "Create Web Service"
   - Wait 10-15 minutes for first build
   - Your app will be live!

## 🔍 Testing After Deployment

1. **Check health endpoint:**
   ```
   https://your-app.onrender.com/api/health
   ```

2. **Test frontend:**
   ```
   https://your-app.onrender.com
   ```

3. **Test scanning:**
   - Allow camera access
   - Try scanning a product
   - Check browser console for errors

## 🐛 Troubleshooting

**Build fails:**
- Check build logs in Render dashboard
- Ensure all files are committed to Git
- Verify `requirements_resnet.txt` is correct

**App crashes:**
- Check logs in Render dashboard
- Verify `refs_resnet.pkl` exists
- Check if `products.csv` is present

**Slow performance:**
- Free tier may be slow (shared resources)
- Consider upgrading to Starter plan
- First request after spin-down takes longer

**Camera not working:**
- HTTPS is required for camera access
- Check browser console for errors
- Try different browser (Chrome recommended)

## 📞 Support

If you encounter issues:
1. Check platform logs
2. Verify all files are deployed
3. Test locally first (`python app.py`)
4. Check platform documentation

---

**Recommended:** Start with **Render** - it's the easiest and has a good free tier!

