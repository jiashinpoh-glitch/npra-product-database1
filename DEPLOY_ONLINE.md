# Deploying Your NPRA R&D Intelligence Platform Online

Host your web application online with a shareable URL, team password protection, automated daily KKMNOW updates, and AI-powered functional health benefit discovery using **GitHub + Supabase + Streamlit Community Cloud**.

---

## Step 1: Create a Free Cloud PostgreSQL Database (Supabase)
1. Go to [supabase.com](https://supabase.com), create a free account, and start a **New Project** (select the Singapore `ap-southeast-1` region for lowest latency in Malaysia).
2. Go to **Project Settings → Database → Connection String (URI)** and copy your PostgreSQL connection string:
   `postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres`

---

## Step 2: Push This Folder to a Private GitHub Repository
1. Create a new **Private Repository** on [github.com](https://github.com) (e.g., `npra-rnd-platform`).
2. Upload all files from this ZIP (including `.github/workflows/daily_sync.yml`).
3. In your GitHub repo, go to **Settings → Secrets and variables → Actions → New repository secret**:
   - Name: `DATABASE_URL`
   - Value: Paste your Supabase connection string from Step 1.
4. Go to the **Actions** tab in GitHub and run **Daily NPRA KKMNOW Data Sync** once manually. It will then auto-sync every working day (Mon–Fri at 9:30 AM MYT).

---

## Step 3: Deploy the Web App on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**, select your `npra-rnd-platform` repository, and set **Main file path** to `app.py`.
3. Click **Advanced settings → Secrets** and paste:
   ```toml
   DATABASE_URL = "postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres"
   APP_PASSWORD = "YourTeamSecretPassword123"
   # Optional: Add either key below to unlock AI mapping for ANY custom health benefit/positioning:
   OPENAI_API_KEY = "sk-..."
   # GEMINI_API_KEY = "AIza..."
   ```
4. Click **Deploy!**
