# Deploying to Vercel - Important Limitations

## ⚠️ Critical Warning

**Vercel is NOT recommended for this application** due to:

1. **Function Size Limit**: 50MB (PyTorch alone is ~500MB+)
2. **Timeout Limits**: 
   - Hobby plan: 10 seconds
   - Pro plan: 60 seconds
   - ML inference can take longer
3. **Cold Starts**: Can be 5-10 seconds for large dependencies
4. **Memory Limits**: May not be enough for PyTorch models

**This deployment will likely FAIL or have severe performance issues.**

## ✅ Better Alternatives

- **Render** (Recommended): https://render.com - Free tier, supports Python apps
- **Railway**: https://railway.app - Easy deployment, auto-detects everything
- **Fly.io**: https://fly.io - Good for global deployment

See `DEPLOYMENT.md` for detailed guides on these platforms.

---

## 🚀 If You Still Want to Try Vercel

### Prerequisites

1. **Vercel Pro Plan** (required for larger functions and longer timeouts)
2. **GitHub account** with your repository
3. **Vercel CLI** installed

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

Or use npx:
```bash
npx vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Configure Project

The `vercel.json` file is already created. However, you'll need to modify the approach:

**Option A: Use Vercel Serverless Functions (Limited)**

1. Create `api/` directory structure
2. Split Flask app into smaller functions
3. Use external ML service (not recommended for this app)

**Option B: Use Vercel with External Backend**

1. Deploy frontend to Vercel (static files)
2. Deploy backend to Render/Railway
3. Connect them via API

### Step 4: Deploy

```bash
vercel
```

Follow the prompts:
- Set up and deploy? **Yes**
- Which scope? **Your account**
- Link to existing project? **No**
- Project name? **checkout-mvp**
- Directory? **./**

### Step 5: Handle Dependencies

Vercel will try to install dependencies, but PyTorch will likely fail due to size.

**Workaround (if you must use Vercel):**

1. Use a lighter ML model (not ResNet50)
2. Use external ML API (like Hugging Face Inference API)
3. Deploy only the frontend to Vercel, backend elsewhere

---

## 🔧 Alternative: Hybrid Deployment

### Deploy Frontend to Vercel, Backend to Render

This is the **best approach** if you want to use Vercel:

#### Step 1: Separate Frontend and Backend

1. **Frontend** (Vercel):
   - Static HTML/CSS/JS
   - Calls backend API

2. **Backend** (Render):
   - Flask API
   - Handles ML inference

#### Step 2: Update Frontend API URLs

In `templates/index.html`, change API calls to point to your Render backend:

```javascript
// Change from:
const API_URL = '';

// To:
const API_URL = 'https://your-app.onrender.com';
```

#### Step 3: Deploy Frontend to Vercel

1. Create a `public/` directory with your frontend files
2. Deploy to Vercel:
   ```bash
   vercel
   ```

#### Step 4: Deploy Backend to Render

Follow instructions in `DEPLOYMENT.md` to deploy backend to Render.

#### Step 5: Enable CORS

The backend already has CORS enabled, so it should work.

---

## 📋 Vercel Configuration Files

I've created:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function wrapper (may not work)

## 🐛 Expected Issues

1. **Build fails**: PyTorch too large
2. **Function timeout**: ML inference takes too long
3. **Cold starts**: Very slow first request
4. **Memory errors**: Not enough memory for PyTorch

## 💡 Recommendation

**Don't use Vercel for this project.** Instead:

1. **Use Render** (easiest, free tier available)
2. **Use Railway** (auto-detects, very easy)
3. **Use Fly.io** (good performance)

All of these are better suited for Python ML applications.

---

## 📞 Need Help?

If you encounter issues:
1. Check Vercel logs: `vercel logs`
2. Check function size: `vercel inspect`
3. Consider switching to Render (see `DEPLOYMENT.md`)

**For a production-ready deployment, use Render or Railway instead of Vercel.**

