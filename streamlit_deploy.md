# Streamlit UI Deployment Guide

## Lokal Körning

### 1. Starta API:et (i en terminal)
```powershell
python run_api.py
```

### 2. Starta Streamlit UI (i en annan terminal)
```powershell
python run_streamlit.py
```

Eller direkt:
```powershell
streamlit run app/ui/streamlit_app.py
```

Öppna: http://localhost:8501

## Deployment på Render.com

### Alternativ 1: Streamlit Cloud (Enklast)

1. Push till GitHub
2. Gå till https://streamlit.io/cloud
3. Connect repository
4. Deploy!

**Streamlit Cloud konfigurerar automatiskt:**
- Port 8501
- Public URL
- Auto-deploy vid push

### Alternativ 2: Render.com (Mer kontroll)

#### Streamlit Service

1. **Skapa `render.yaml`:**
```yaml
services:
  - type: web
    name: avarn-streamlit
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: API_URL
        value: https://avarn-api.onrender.com  # Din FastAPI URL
```

2. **Deploy på Render:**
   - Connect GitHub repo
   - Render läser `render.yaml` automatiskt
   - Streamlit körs på port från `$PORT`

#### FastAPI Service (för backend)

1. **Skapa separat service för API:**
```yaml
services:
  - type: web
    name: avarn-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ENVIRONMENT
        value: production
```

### Alternativ 3: Docker (Fullständig kontroll)

**Dockerfile för Streamlit:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8501

CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    # ... (från tidigare)

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

## Environment Variables

För production, sätt:
```bash
API_URL=https://din-api-url.com
```

I Streamlit Cloud:
- Settings → Secrets → Add secret
- Key: `API_URL`
- Value: `https://din-api-url.com`

## Tips

1. **Separera API och UI:**
   - API: Render/Heroku
   - UI: Streamlit Cloud (gratis!)

2. **CORS:**
   - Om API och UI på olika domäner, lägg till CORS i FastAPI

3. **Caching:**
   - Streamlit cachar automatiskt med `@st.cache_data`

4. **Secrets:**
   - Använd `st.secrets` för API_URL i production

## Quick Deploy Checklist

- [ ] API deployad och fungerar
- [ ] API_URL satt i Streamlit secrets
- [ ] GitHub repo uppdaterat
- [ ] Streamlit Cloud/Render konfigurerat
- [ ] Testat lokalt först

---

**Streamlit är perfekt för ML-UI!** 🚀
