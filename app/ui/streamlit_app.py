"""
Streamlit UI for Fire Alarm Testing Price Prediction.
Production-ready web interface.
"""

import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Avarn - Brandlarmstestning Prisprediktion",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL (can be configured via environment variable or secrets)
import os

# Default API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Try to get from Streamlit secrets (if available and file exists)
# This handles the case where secrets.toml doesn't exist gracefully
# StreamlitSecretNotFoundError is raised when secrets.toml doesn't exist
try:
    # Try to access secrets - this will raise StreamlitSecretNotFoundError if file doesn't exist
    # We catch all exceptions to handle missing secrets file gracefully
    api_url_from_secrets = st.secrets.get("API_URL", None)
    if api_url_from_secrets:
        API_URL = api_url_from_secrets
except Exception:
    # Secrets file doesn't exist or other error, use environment variable or default
    # This is expected in local development without secrets.toml
    pass

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None


def make_prediction(data):
    """Make prediction via API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🔥 Avarn - Brandlarmstestning Prisprediktion</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Inställningar")
        
        # API Health Check
        api_healthy, health_data = check_api_health()
        if api_healthy:
            st.success("✅ API är online")
            if health_data:
                st.caption(f"Model: {health_data.get('model_version', 'N/A')}")
                st.caption(f"Pipeline: {health_data.get('feature_pipeline_version', 'N/A')}")
        else:
            st.error("❌ API är offline")
            st.info("Kontrollera att API:et körs på http://localhost:8000")
        
        st.divider()
        
        st.header("📊 Information")
        st.markdown("""
        Detta system predikterar priser för brandlarmstestning baserat på:
        - Antal sektioner
        - Antal detektorer
        - Antal larmdon
        - Testningsfrekvens
        - Plats
        - Ytterligare utrustning
        """)
        
        st.divider()
        
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Powered by:** MLflow + FastAPI + Streamlit")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Prediktion", "📊 Batch Prediktion", "ℹ️ Om Systemet"])
    
    with tab1:
        st.header("Enskild Prisprediktion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Systeminformation")
            
            antal_sektioner = st.number_input(
                "Antal Sektioner",
                min_value=1,
                max_value=50,
                value=8,
                help="Antal brandlarmsektioner i systemet"
            )
            
            antal_detektorer = st.number_input(
                "Antal Detektorer",
                min_value=1,
                max_value=200,
                value=25,
                help="Totalt antal branddetektorer"
            )
            
            antal_larmdon = st.number_input(
                "Antal Larmdon",
                min_value=1,
                max_value=100,
                value=15,
                help="Antal larmdon (sirener, klockor, etc.)"
            )
            
            dörrhållarmagneter = st.number_input(
                "Dörrhållarmagneter",
                min_value=0,
                max_value=50,
                value=5,
                help="Antal dörrhållarmagneter"
            )
        
        with col2:
            st.subheader("Plats & Testning")
            
            stad = st.selectbox(
                "Stad",
                ["Stockholm", "Göteborg", "Malmö", "Uppsala", "Linköping", "Örebro", "Västerås", "Helsingborg"],
                help="Plats för testningen"
            )
            
            ventilation = st.radio(
                "Ventilation",
                [0, 1],
                format_func=lambda x: "Ja" if x == 1 else "Nej",
                help="Har byggnaden ventilationssystem?"
            )
            
            st.subheader("Testningsfrekvens")
            frequency = st.radio(
                "Välj frekvens",
                ["kvartalsvis", "månadsvis", "årsvis"],
                help="Hur ofta ska testningen utföras?"
            )
            
            kvartalsvis = 1 if frequency == "kvartalsvis" else 0
            månadsvis = 1 if frequency == "månadsvis" else 0
            årsvis = 1 if frequency == "årsvis" else 0
        
        # Prediction button
        if st.button("🎯 Prediktera Pris", type="primary", use_container_width=True):
            if not api_healthy:
                st.error("API är inte tillgängligt. Starta API:et först.")
            else:
                with st.spinner("Beräknar prediktion..."):
                    request_data = {
                        "antal_sektioner": int(antal_sektioner),
                        "antal_detektorer": int(antal_detektorer),
                        "antal_larmdon": int(antal_larmdon),
                        "dörrhållarmagneter": int(dörrhållarmagneter),
                        "ventilation": int(ventilation),
                        "stad": stad,
                        "kvartalsvis": kvartalsvis,
                        "månadsvis": månadsvis,
                        "årsvis": årsvis
                    }
                    
                    success, result = make_prediction(request_data)
                    
                    if success:
                        predicted_price = result.get("predicted_price", 0)
                        confidence_lower = result.get("confidence_interval_lower", 0)
                        confidence_upper = result.get("confidence_interval_upper", 0)
                        prediction_id = result.get("prediction_id", "N/A")
                        
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.metric(
                            "Predikterat Pris",
                            f"{predicted_price:,.0f} SEK",
                            delta=f"±{(confidence_upper - confidence_lower)/2:,.0f} SEK"
                        )
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Nedre gräns", f"{confidence_lower:,.0f} SEK")
                        with col_b:
                            st.metric("Övre gräns", f"{confidence_upper:,.0f} SEK")
                        with col_c:
                            st.metric("Konfidensintervall", f"±{((confidence_upper - confidence_lower)/2):,.0f} SEK")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Model info
                        with st.expander("ℹ️ Modellinformation"):
                            st.write(f"**Modellversion:** {result.get('model_version', 'N/A')}")
                            st.write(f"**Pipeline-version:** {result.get('feature_pipeline_version', 'N/A')}")
                            st.write(f"**Prediktion ID:** {prediction_id}")
                        
                        st.success("✅ Prediktion genomförd!")
                    else:
                        error_msg = result.get("detail", result.get("error", "Okänt fel"))
                        st.error(f"❌ Fel: {error_msg}")
    
    with tab2:
        st.header("Batch Prediktion")
        st.info("Ladda upp en CSV-fil med flera objekt för batch-prediktion.")
        
        uploaded_file = st.file_uploader(
            "Välj CSV-fil",
            type=["csv"],
            help="CSV-fil med kolumner: antal_sektioner, antal_detektorer, antal_larmdon, dörrhållarmagneter, ventilation, stad, kvartalsvis, månadsvis, årsvis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df, use_container_width=True)
                
                if st.button("🎯 Prediktera Alla", type="primary"):
                    if not api_healthy:
                        st.error("API är inte tillgängligt.")
                    else:
                        # Convert to request format
                        items = []
                        for _, row in df.iterrows():
                            items.append({
                                "antal_sektioner": int(row.get("antal_sektioner", 0)),
                                "antal_detektorer": int(row.get("antal_detektorer", 0)),
                                "antal_larmdon": int(row.get("antal_larmdon", 0)),
                                "dörrhållarmagneter": int(row.get("dörrhållarmagneter", 0)),
                                "ventilation": int(row.get("ventilation", 0)),
                                "stad": str(row.get("stad", "Stockholm")),
                                "kvartalsvis": int(row.get("kvartalsvis", 0)),
                                "månadsvis": int(row.get("månadsvis", 0)),
                                "årsvis": int(row.get("årsvis", 0))
                            })
                        
                        with st.spinner("Bearbetar batch-prediktion..."):
                            try:
                                response = requests.post(
                                    f"{API_URL}/predict/batch",
                                    json={"items": items},
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    predictions = result.get("predictions", [])
                                    
                                    # Create results DataFrame
                                    results_df = df.copy()
                                    results_df["predikterat_pris"] = [p["predicted_price"] for p in predictions]
                                    results_df["nedre_gräns"] = [p.get("confidence_interval_lower", 0) for p in predictions]
                                    results_df["övre_gräns"] = [p.get("confidence_interval_upper", 0) for p in predictions]
                                    
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Download button
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Ladda ner resultat (CSV)",
                                        csv,
                                        "predictions.csv",
                                        "text/csv"
                                    )
                                    
                                    st.success(f"✅ {len(predictions)} prediktioner genomförda!")
                                else:
                                    st.error(f"Fel: {response.json()}")
                            except Exception as e:
                                st.error(f"Fel vid batch-prediktion: {str(e)}")
            except Exception as e:
                st.error(f"Fel vid läsning av fil: {str(e)}")
    
    with tab3:
        st.header("Om Systemet")
        
        st.markdown("""
        ### 🏗️ Systemarkitektur
        
        Detta system består av:
        
        1. **ML Model** - Gradient Boosting Regressor
           - Tränad på historisk data
           - Test R²: 98.61%
           - Test RMSE: 3,158 SEK
        
        2. **FastAPI Backend** - REST API för inference
           - Validering med Pydantic
           - Automatisk logging
           - Health checks
        
        3. **Streamlit Frontend** - Användargränssnitt
           - Interaktiv prediktion
           - Batch-processing
           - Real-time feedback
        
        4. **MLflow** - Model versionering & tracking
           - Experiment tracking
           - Model registry
           - Metrics & parameters
        
        ### 📊 Features
        
        - ✅ Enskild prediktion
        - ✅ Batch prediktion
        - ✅ Konfidensintervall
        - ✅ Model versionering
        - ✅ Prediction logging
        
        ### 🚀 Deployment
        
        Systemet kan deployas på:
        - Render.com
        - Streamlit Cloud
        - Heroku
        - AWS/GCP/Azure
        """)
        
        st.divider()
        
        st.subheader("📈 Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", "0.9861")
        with col2:
            st.metric("Test RMSE", "3,159 SEK")
        with col3:
            st.metric("Test MAE", "1,997 SEK")


if __name__ == "__main__":
    main()
