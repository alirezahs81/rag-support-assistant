# Deploy live demo on Hugging Face Spaces

Primary host for the public demo (alternative to Streamlit Community Cloud).

## Create the Space

1. Open [https://huggingface.co/new-space](https://huggingface.co/new-space) (sign in with HF account; you can link GitHub).
2. **Space name:** e.g. `rag-support-assistant`
3. **License:** MIT
4. **SDK:** **Streamlit**
5. **Space hardware:** CPU basic (free)
6. Create the Space.

## Connect this GitHub repo

**Option A — GitHub sync (recommended)**

1. In the Space: **Settings → Repository → Connect to GitHub** (or “Factory reboot” / linked repo, depending on UI).
2. Select `alirezahs81/rag-support-assistant`, branch `main`.
3. Entrypoint / app file: `app.py`
4. Wait for the build (installs `requirements.txt`, runs Streamlit).

**Option B — manual push**

```bash
git clone https://huggingface.co/spaces/<your-username>/rag-support-assistant
cd rag-support-assistant
# copy app.py, src/, data/, requirements.txt from GitHub
git add -A && git commit -m "Initial Space" && git push
```

## After it is live

Public URL shape:

`https://huggingface.co/spaces/<your-username>/rag-support-assistant`

Embed / direct app:

`https://<your-username>-rag-support-assistant.hf.space`

Add that link to your [GitHub profile README](https://github.com/alirezahs81/alirezahs81).

## Notes

- No API keys required.
- First load builds the TF-IDF index once (cached).
- Free Spaces may sleep when idle; the next visit wakes them (similar to other free hosts).
- If the build fails, open the Space **Logs** tab and check Python errors.
