# Deploy the Streamlit demo

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Open [https://share.streamlit.io](https://share.streamlit.io).
2. Sign in with the GitHub account that owns `alirezahs81/rag-support-assistant`.
3. Click **Create app** / **New app**.
4. Fill in:
   - **Repository:** `alirezahs81/rag-support-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**.
6. Wait until the app status is **Running** (usually a few minutes).
7. Copy the public URL, e.g. `https://rag-support-assistant-….streamlit.app`.

### After deploy

- Add the URL to your [profile README](https://github.com/alirezahs81/alirezahs81) under Featured projects.
- Optional: in the Streamlit app settings, enable **Ready** / public access so recruiters need no login.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Module not found `src` | `app.py` already adds the repo root to `sys.path` |
| Cold start slow | First load builds the TF-IDF index once (`st.cache_resource`) |
| Dependency error | Ensure `requirements.txt` on `main` includes `streamlit` and `scikit-learn` |

No secrets or API keys are required for this demo.
