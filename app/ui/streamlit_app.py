"""
Streamlit UI for Fire Alarm Testing Price Prediction.
Production-ready web interface.
"""

import pandas as pd
import requests
import streamlit as st

from app.config import Config

# Page config
st.set_page_config(
    page_title="Brandlarmsavtal Prisprediktion",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get API URL from config
API_URL = Config.STREAMLIT_API_URL

# Custom CSS with new color scheme
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background-color: #232323;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #232323;
        color: white;
    }
    
    /* Sidebar - comprehensive styling */
    section[data-testid="stSidebar"],
    .css-1d391kg,
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] nav,
    .css-1lcbmhc,
    .css-1y4p8pa {
        background-color: #232323 !important;
        color: white !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* Sidebar success/error/info */
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stError,
    [data-testid="stSidebar"] .stInfo {
        background-color: #2D2D2D !important;
        color: white !important;
    }
    
    /* Cards and containers */
    .prediction-box, .metric-box, 
    div[data-testid="stMetricValue"],
    div[data-testid="stDataFrame"],
    .element-container {
        background-color: #2D2D2D;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Systeminformation card - wrap the entire column content */
    .system-info-card {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Plats & Testning card */
    .location-testing-card {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Target the first column (col1) which contains Systeminformation */
    .stColumn:first-child > div,
    div[data-testid="column"]:first-child > div {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Target the second column (col2) which contains Plats & Testning */
    .stColumn:nth-child(2) > div,
    div[data-testid="column"]:nth-child(2) > div {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Ensure all elements inside cards have proper spacing */
    .system-info-card > *,
    .location-testing-card > * {
        margin-bottom: 0.5rem;
    }
    
    .system-info-card > *:last-child,
    .location-testing-card > *:last-child {
        margin-bottom: 0;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 1rem;
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Hide empty container above header - comprehensive targeting */
    .main .block-container > div:first-child:empty,
    .main .block-container > div:first-child:has(> div:empty),
    .main .element-container:first-child:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }
    
    /* Hide the first element container if it doesn't contain the header */
    .main .block-container > div:first-child:not(:has(.main-header)) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Hide empty markdown containers */
    .main .block-container > div:first-child .stMarkdown:empty,
    .stMarkdown:empty,
    div[data-testid="stMarkdownContainer"]:empty,
    div[data-testid="stMarkdownContainer"]:has(> div:empty) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove top padding from main container */
    .main .block-container {
        padding-top: 1rem !important;
    }
    
    /* Target the header container specifically - remove top margin */
    .main .block-container > div:has(.main-header) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Hide any Streamlit containers before the header - aggressive targeting */
    .main .block-container > div:first-child:not(:has(.main-header)) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    
    /* Alternative: Hide first child if it's not the header container */
    .main .block-container > div:nth-child(1):not(:has(.main-header)) {
        display: none !important;
    }
    
    /* Make sure header is the first visible element */
    .main .block-container > div:has(.main-header) {
        order: -1;
    }
    
    /* Prediction box */
    .prediction-box {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #58C5CA;
        color: white;
    }
    
    /* Metric box */
    .metric-box {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: white;
    }
    
    /* Primary buttons */
    .stButton > button {
        background-color: #58C5CA;
        color: black;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #4ab5ba;
        color: black;
    }
    
    /* Secondary buttons (text buttons) */
    button[kind="secondary"],
    .stDownloadButton > button {
        background-color: transparent;
        color: black;
        border: 1px solid #58C5CA;
    }
    
    button[kind="secondary"]:hover,
    .stDownloadButton > button:hover {
        background-color: #58C5CA;
        color: black;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background-color: #58C5CA;
        color: black;
        border: none;
    }
    
    .stDownloadButton > button:hover {
        background-color: #4ab5ba;
        color: black;
    }
    
    /* All text white */
    p, h1, h2, h3, h4, h5, h6, label, span, div {
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #2D2D2D;
        color: white;
        border: 1px solid #58C5CA;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2D2D2D;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #2D2D2D;
        color: white;
    }
    
    /* Success/Error/Info messages */
    .stSuccess {
        background-color: #2D2D2D;
        border-left: 4px solid #58C5CA;
    }
    
    .stError {
        background-color: #2D2D2D;
    }
    
    .stInfo {
        background-color: #2D2D2D;
    }
    
    /* Dividers */
    hr {
        border-color: #58C5CA;
    }
    
    /* Selectbox and dropdowns */
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label {
        color: white !important;
    }
    
    /* Additional sidebar styling - force dark background */
    div[data-testid="stSidebar"],
    div[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebar"] > div:first-child > div:first-child {
        background-color: #232323 !important;
        background-image: none !important;
    }
    
    /* Sidebar content containers */
    div[data-testid="stSidebar"] .element-container,
    div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] .stHeader {
        background-color: transparent !important;
        color: white !important;
    }
    
    /* Override any light backgrounds in sidebar */
    div[data-testid="stSidebar"] * {
        background-color: transparent !important;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #232323 0%, #232323 100%) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, (
            response.json() if response.status_code == 200 else None
        )
    except:
        return False, None


def make_prediction(data):
    """Make prediction via API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def main():
    """Main Streamlit app."""

    # Header - using write to avoid empty container
    st.markdown(
        '<div class="main-header">Brandlarmsavtal FDP</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:

        st.markdown("""
        ## Information
        Detta system predikterar priser för avtal på brandlarmsanläggningar där Schneider FDP är installerat baserat på:
        - Antal sektioner
        - Antal detektorer
        - Antal larmdon
        - Testningsfrekvens
        - Plats
        - Ytterligare utrustning
        """)

                # API Health Check
        api_healthy, health_data = check_api_health()
        if api_healthy:
            "✅ API är online"
        else:
            "❌ API är offline"
            "Kontrollera att API:et körs på http://localhost:8000"

    # Main content
    tab1, tab2, tab3 = st.tabs(
        ["Manuell inmating", "Filuppladdning", "Om systemet"]
    )

    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            # Systeminformation card
            with st.container():
                st.markdown(
                    '<div class="system-info-card">',
                    unsafe_allow_html=True,
                )
                st.subheader("Systeminformation")

                antal_sektioner = st.number_input(
                    "Antal Sektioner",
                    min_value=1,
                    max_value=50,
                    value=8,
                    help="Antal brandlarmsektioner i systemet",
                )

                antal_detektorer = st.number_input(
                    "Antal Detektorer",
                    min_value=1,
                    max_value=200,
                    value=25,
                    help="Totalt antal branddetektorer",
                )

                antal_larmdon = st.number_input(
                    "Antal Larmdon",
                    min_value=1,
                    max_value=100,
                    value=15,
                    help="Antal larmdon (sirener, klockor, etc.)",
                )

                dörrhållarmagneter = st.number_input(
                    "Dörrhållarmagneter",
                    min_value=0,
                    max_value=50,
                    value=5,
                    help="Antal dörrhållarmagneter",
                )
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Plats & Testning card
            with st.container():
                st.markdown(
                    '<div class="location-testing-card">',
                    unsafe_allow_html=True,
                )
                st.subheader("Plats & Testning")

                stad = st.selectbox(
                    "Stad",
                    [
                        "Stockholm",
                        "Göteborg",
                        "Malmö",
                        "Uppsala",
                        "Linköping",
                        "Örebro",
                        "Västerås",
                        "Helsingborg",
                    ],
                    help="Plats för testningen",
                )

                ventilation = st.radio(
                    "Ventilation",
                    [0, 1],
                    format_func=lambda x: "Ja" if x == 1 else "Nej",
                    help="Har byggnaden ventilationssystem?",
                )

                st.subheader("Testningsfrekvens")
                frequency = st.radio(
                    "Välj frekvens",
                    ["kvartalsvis", "månadsvis", "årsvis"],
                    help="Hur ofta ska testningen utföras?",
                )

                kvartalsvis = 1 if frequency == "kvartalsvis" else 0
                månadsvis = 1 if frequency == "månadsvis" else 0
                årsvis = 1 if frequency == "årsvis" else 0
                st.markdown('</div>', unsafe_allow_html=True)

        # Prediction button
        if st.button("Prediktera Pris", type="primary", use_container_width=True):
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
                        "årsvis": årsvis,
                    }

                    success, result = make_prediction(request_data)

                    if success:
                        predicted_price = result.get("predicted_price", 0)
                        confidence_lower = result.get("confidence_interval_lower", 0)
                        confidence_upper = result.get("confidence_interval_upper", 0)
                        prediction_id = result.get("prediction_id", "N/A")

                        # Display prediction
                        st.markdown(
                            '<div class="prediction-box">', unsafe_allow_html=True
                        )
                        st.metric(
                            "Predikterat Pris",
                            f"{predicted_price:,.0f} SEK",
                            delta=f"±{(confidence_upper - confidence_lower)/2:,.0f} SEK",
                        )

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Nedre gräns", f"{confidence_lower:,.0f} SEK")
                        with col_b:
                            st.metric("Övre gräns", f"{confidence_upper:,.0f} SEK")
                        with col_c:
                            st.metric(
                                "Konfidensintervall",
                                f"±{((confidence_upper - confidence_lower)/2):,.0f} SEK",
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

                        # Model info
                        with st.expander("ℹ️ Modellinformation"):
                            st.write(
                                f"**Modellversion:** {result.get('model_version', 'N/A')}"
                            )
                            st.write(
                                f"**Pipeline-version:** {result.get('feature_pipeline_version', 'N/A')}"
                            )
                            st.write(f"**Prediktion ID:** {prediction_id}")

                        st.success("✅ Prediktion genomförd!")
                    else:
                        error_msg = result.get(
                            "detail", result.get("error", "Okänt fel")
                        )
                        st.error(f"❌ Fel: {error_msg}")

    with tab2:
        st.info("Ladda upp en Config-fil prediktion.")

        uploaded_file = st.file_uploader(
            "Välj Config-fil",
            type=["json"],
            help="CSV-fil med kolumner: antal_sektioner, antal_detektorer, antal_larmdon, dörrhållarmagneter, ventilation, stad, kvartalsvis, månadsvis, årsvis",
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
                            items.append(
                                {
                                    "antal_sektioner": int(
                                        row.get("antal_sektioner", 0)
                                    ),
                                    "antal_detektorer": int(
                                        row.get("antal_detektorer", 0)
                                    ),
                                    "antal_larmdon": int(row.get("antal_larmdon", 0)),
                                    "dörrhållarmagneter": int(
                                        row.get("dörrhållarmagneter", 0)
                                    ),
                                    "ventilation": int(row.get("ventilation", 0)),
                                    "stad": str(row.get("stad", "Stockholm")),
                                    "kvartalsvis": int(row.get("kvartalsvis", 0)),
                                    "månadsvis": int(row.get("månadsvis", 0)),
                                    "årsvis": int(row.get("årsvis", 0)),
                                }
                            )

                        with st.spinner("Bearbetar batch-prediktion..."):
                            try:
                                response = requests.post(
                                    f"{API_URL}/predict/batch",
                                    json={"items": items},
                                    timeout=30,
                                )

                                if response.status_code == 200:
                                    result = response.json()
                                    predictions = result.get("predictions", [])

                                    # Create results DataFrame
                                    results_df = df.copy()
                                    results_df["predikterat_pris"] = [
                                        p["predicted_price"] for p in predictions
                                    ]
                                    results_df["nedre_gräns"] = [
                                        p.get("confidence_interval_lower", 0)
                                        for p in predictions
                                    ]
                                    results_df["övre_gräns"] = [
                                        p.get("confidence_interval_upper", 0)
                                        for p in predictions
                                    ]

                                    st.dataframe(results_df, use_container_width=True)

                                    # Download button
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Ladda ner resultat (CSV)",
                                        csv,
                                        "predictions.csv",
                                        "text/csv",
                                    )

                                    st.success(
                                        f"✅ {len(predictions)} prediktioner genomförda!"
                                    )
                                else:
                                    st.error(f"Fel: {response.json()}")
                            except Exception as e:
                                st.error(f"Fel vid batch-prediktion: {str(e)}")
            except Exception as e:
                st.error(f"Fel vid läsning av fil: {str(e)}")

    with tab3:

        st.markdown("""
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
        
        ### Features
        
        - Enskild prediktion
        - Batch prediktion
        - Konfidensintervall
        - Model versionering
        - Prediction logging
        
        ### Deployment
        
        Systemet kan deployas på:
        - Render.com
        - Streamlit Cloud
        - Heroku
        - AWS/GCP/Azure
        """)

        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", "0.9861")
        with col2:
            st.metric("Test RMSE", "3,159 SEK")
        with col3:
            st.metric("Test MAE", "1,997 SEK")


if __name__ == "__main__":
    main()
