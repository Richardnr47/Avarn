# Streamlit UI Guide

## 🚀 Snabbstart

### 1. Starta API:et (Terminal 1)
```powershell
python run_api.py
```

### 2. Starta Streamlit UI (Terminal 2)
```powershell
python run_streamlit.py
```

### 3. Öppna i webbläsare
**http://localhost:8501**

## 📋 Funktioner

### Enskild Prediktion
- Fyll i formulär med systeminformation
- Få omedelbar pris-prediktion
- Se konfidensintervall
- Modellinformation

### Batch Prediktion
- Ladda upp CSV-fil med flera objekt
- Prediktera alla samtidigt
- Ladda ner resultat som CSV

### Systeminformation
- Model performance metrics
- Systemarkitektur
- Deployment info

## ⚙️ Konfiguration

### Lokal Utveckling
API URL sätts automatiskt till `http://localhost:8000`

### Production
Skapa `.streamlit/secrets.toml`:
```toml
API_URL = "https://din-api-url.com"
```

Eller sätt environment variable:
```bash
export API_URL="https://din-api-url.com"
```

## 🚀 Deployment

### Streamlit Cloud (Rekommenderat - Gratis!)

1. **Push till GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit UI"
   git push
   ```

2. **Deploy på Streamlit Cloud**
   - Gå till https://streamlit.io/cloud
   - Sign in med GitHub
   - Klicka "New app"
   - Välj repository och branch
   - Main file: `app/ui/streamlit_app.py`
   - Klicka "Deploy"

3. **Konfigurera Secrets**
   - Settings → Secrets
   - Lägg till:
     ```toml
     API_URL = "https://din-api-url.com"
     ```

### Render.com

Använd `render.yaml` (redan skapat):
- Streamlit service konfigurerad
- API service konfigurerad
- Auto-deploy vid push

### Docker

```bash
docker build -f Dockerfile.streamlit -t avarn-streamlit .
docker run -p 8501:8501 -e API_URL=http://api:8000 avarn-streamlit
```

## 🔧 Troubleshooting

### "API är offline"
- Kontrollera att API:et körs på port 8000
- Kontrollera API_URL i secrets.toml eller environment

### "No secrets found"
- Detta är OK i lokal utveckling
- Appen använder default `http://localhost:8000`
- För production, skapa `.streamlit/secrets.toml`

### Port redan använd
```powershell
# Ändra port i run_streamlit.py eller:
streamlit run app/ui/streamlit_app.py --server.port 8502
```

## 📊 Features

✅ Responsive design
✅ Real-time predictions
✅ Batch processing
✅ Error handling
✅ Health checks
✅ Model versioning info

---

**Streamlit är perfekt för ML-UI!** 🎉
