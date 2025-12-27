# Vercel Deployment Guide

## ⚠️ IMPORTANT: Vercel Limitations

**Vercel is NOT ideal for this application** because:

1. **Function Size Limit**: 50MB (PyTorch is ~500MB+)
2. **Timeout Limits**: 
   - Hobby: 10 seconds
   - Pro: 60 seconds
   - ML inference may exceed this
3. **Cold Starts**: 5-10 seconds for large dependencies
4. **Memory**: May not be sufficient for PyTorch

**This will likely FAIL or have severe performance issues.**

## ✅ Recommended: Use Render Instead

**Render** is much better for Python ML apps:
- Free tier available
- No function size limits
- Supports long-running processes
- Easy deployment

**Quick Deploy to Render:**
1. Go to https://render.com
2. Connect your GitHub repo
3. Select "Web Service"
4. Build: `pip install -r requirements_resnet.txt`
5. Start: `python app.py`
6. Done!

See `DEPLOYMENT.md` for detailed Render instructions.

---

## 🚀 If You Still Want to Try Vercel

### Option 1: Hybrid Deployment (Recommended if using Vercel)

**Deploy frontend to Vercel, backend to Render**

This is the best approach if you want to use Vercel:

#### Step 1: Deploy Backend to Render

1. Go to https://render.com
2. Connect your GitHub repo: https://github.com/AbdullahYaseen01/sda
3. Create new Web Service
4. Configure:
   - **Name**: `checkout-mvp-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements_resnet.txt`
   - **Start Command**: `python app.py`
5. Deploy and note your URL: `https://checkout-mvp-backend.onrender.com`

#### Step 2: Update Frontend for Vercel

Create a new frontend-only version:

1. **Create `public/index.html`** (copy from `templates/index.html`)
2. **Update API URLs** to point to Render backend
3. **Deploy to Vercel**

#### Step 3: Deploy Frontend to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel
```

---

### Option 2: Full Vercel Deployment (Will Likely Fail)

If you want to try deploying everything to Vercel:

#### Prerequisites

- Vercel Pro Plan (required for larger functions)
- Node.js installed
- Vercel CLI

#### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

#### Step 2: Login

```bash
vercel login
```

#### Step 3: Deploy

```bash
vercel
```

Follow prompts:
- Set up and deploy? **Yes**
- Which scope? **Your account**
- Link to existing project? **No**
- Project name? **checkout-mvp**
- Directory? **./**

#### Step 4: Configure Environment Variables

In Vercel dashboard:
- Go to Project Settings → Environment Variables
- Add: `PYTHON_VERSION=3.11`

#### Expected Issues

1. **Build fails**: PyTorch exceeds 50MB limit
2. **Function timeout**: ML inference too slow
3. **Memory errors**: Not enough for PyTorch

---

## 📁 Files Created for Vercel

I've created:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function wrapper (may not work)

## 🔧 Alternative: Lightweight Vercel Function

If you must use Vercel, consider:

1. **Use external ML API** (Hugging Face, Google Cloud Vision)
2. **Use lighter model** (not ResNet50)
3. **Split into microservices**

But this requires significant code changes.

---

## 💡 My Strong Recommendation

**Don't use Vercel for this project.**

Instead, use:
1. **Render** (easiest, free tier) ← **BEST CHOICE**
2. **Railway** (auto-detects, very easy)
3. **Fly.io** (good performance)

All are better suited for Python ML applications.

---

## 📋 Quick Comparison

| Platform | Free Tier | Python Support | ML Support | Ease |
|----------|-----------|----------------|------------|------|
| **Render** | ✅ Yes | ✅ Excellent | ✅ Perfect | ⭐⭐⭐⭐⭐ |
| **Railway** | ✅ Yes | ✅ Excellent | ✅ Perfect | ⭐⭐⭐⭐⭐ |
| **Vercel** | ⚠️ Limited | ⚠️ Serverless only | ❌ Too small | ⭐⭐ |

---

## 🚀 Recommended: Deploy to Render (5 minutes)

1. **Go to**: https://render.com
2. **Sign up** (free with GitHub)
3. **Click**: "New +" → "Web Service"
4. **Connect**: Your GitHub repo (https://github.com/AbdullahYaseen01/sda)
5. **Configure**:
   - Name: `checkout-mvp`
   - Environment: `Python 3`
   - Build: `pip install -r requirements_resnet.txt`
   - Start: `python app.py`
6. **Deploy** - Done!

Your app will be live at: `https://checkout-mvp.onrender.com`

---

## 📞 Need Help?

- **Vercel issues**: Check `vercel logs`
- **Better option**: See `DEPLOYMENT.md` for Render/Railway guides
- **Questions**: Check platform documentation

**For production use, choose Render or Railway over Vercel.**

