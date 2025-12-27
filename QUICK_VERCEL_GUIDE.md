# Quick Vercel Deployment Steps

## ⚠️ Warning: Vercel Has Limitations

**This app uses PyTorch (~500MB) which exceeds Vercel's 50MB function limit.**

**Recommended**: Use **Render** instead (see `DEPLOYMENT.md`)

---

## 🚀 Step-by-Step: Deploy to Vercel

### Prerequisites

1. **Vercel account** (sign up at https://vercel.com)
2. **Vercel CLI** installed
3. **Node.js** installed (for Vercel CLI)

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

This will open your browser for authentication.

### Step 3: Navigate to Project

```bash
cd d:\haris\checkout_mvp
```

### Step 4: Deploy

```bash
vercel
```

Follow the prompts:
- **Set up and deploy?** → Type `Y` and press Enter
- **Which scope?** → Select your account
- **Link to existing project?** → Type `N` and press Enter
- **Project name?** → Type `checkout-mvp` and press Enter
- **Directory?** → Press Enter (use current directory `./`)

### Step 5: Production Deploy

After initial deployment, deploy to production:

```bash
vercel --prod
```

---

## 🔧 Configuration

The `vercel.json` file is already configured. It includes:
- Python 3.11 runtime
- Flask app routing
- API endpoints

## ⚠️ Expected Issues

1. **Build will likely fail** due to PyTorch size (>50MB limit)
2. **If it builds**, functions may timeout during ML inference
3. **Cold starts** will be very slow (5-10 seconds)

## 💡 Better Solution: Hybrid Deployment

**Deploy frontend to Vercel, backend to Render:**

### Part 1: Deploy Backend to Render

1. Go to https://render.com
2. Connect GitHub repo: https://github.com/AbdullahYaseen01/sda
3. Create Web Service:
   - Build: `pip install -r requirements_resnet.txt`
   - Start: `python app.py`
4. Get your backend URL: `https://your-app.onrender.com`

### Part 2: Update Frontend

Modify `templates/index.html` to use your Render backend URL instead of relative paths.

### Part 3: Deploy Frontend to Vercel

Deploy only the frontend files to Vercel.

---

## ✅ Recommended: Use Render Instead

**Much easier and better suited for this app:**

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect repo: https://github.com/AbdullahYaseen01/sda
5. Configure:
   - Build: `pip install -r requirements_resnet.txt`
   - Start: `python app.py`
6. Deploy!

**Takes 5 minutes, works perfectly!**

See `DEPLOYMENT.md` for detailed Render instructions.

---

## 📞 Troubleshooting

**Vercel build fails:**
- PyTorch is too large (expected)
- Switch to Render instead

**Function timeout:**
- ML inference takes too long
- Vercel Pro plan has 60s limit (may still be too short)

**Memory errors:**
- Not enough memory for PyTorch
- Use Render or Railway instead

---

## 🎯 Final Recommendation

**Don't use Vercel for this project.**

Use **Render** - it's:
- ✅ Free tier available
- ✅ No size limits
- ✅ Perfect for Python ML apps
- ✅ Easy deployment
- ✅ Better performance

**Deploy to Render in 5 minutes - see `DEPLOYMENT.md`**

