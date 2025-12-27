# Quick Deployment Guide

## ⚠️ Netlify Won't Work

Netlify is for static websites only. Your app needs:
- A Python server running continuously
- PyTorch (500MB+ dependencies)
- Real-time ML inference

**Use Render instead** - it's free and perfect for Python apps!

## 🚀 Deploy to Render (5 Minutes)

### Step 1: Prepare Your Code

1. **Make sure all files are ready:**
   ```bash
   # Verify these files exist:
   - app.py
   - requirements_resnet.txt (or requirements.txt)
   - refs_resnet.pkl
   - products.csv
   - products/ directory
   - templates/index.html
   ```

2. **Push to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Ready for deployment"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. **Go to:** https://render.com
2. **Sign up** (free with GitHub)
3. **Click:** "New +" → "Web Service"
4. **Connect** your GitHub repository
5. **Configure:**
   - Name: `checkout-mvp`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements_resnet.txt`
   - Start Command: `python app.py`
6. **Click:** "Create Web Service"
7. **Wait** 10-15 minutes for first build
8. **Done!** Your app is live at `https://checkout-mvp.onrender.com`

## 📝 Alternative: Railway (Even Easier)

1. **Go to:** https://railway.app
2. **Click:** "New Project" → "Deploy from GitHub"
3. **Select** your repository
4. **Railway auto-detects** everything!
5. **Deploy** - that's it!

## ✅ After Deployment

1. **Test your app:**
   - Visit your deployment URL
   - Check: `https://your-app.onrender.com/api/health`

2. **Test scanning:**
   - Allow camera access
   - Try scanning a product

## 🆘 Need Help?

See `DEPLOYMENT.md` for detailed instructions and troubleshooting.

