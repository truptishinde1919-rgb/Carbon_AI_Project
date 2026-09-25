"""
app.py
-------
STEP 3 of the Carbon AI project — THE WEB APP ITSELF.

WHAT THIS FILE DOES:
Builds an interactive dashboard using Streamlit (a Python library that turns
a plain script into a web app — no HTML/CSS/JS needed). A user picks a
biomass feedstock, sets pyrolysis process conditions with sliders, and the
app instantly predicts the resulting Biochar Yield (%) using the model we
trained in train_model.py. It also shows WHY the model made that prediction
(feature importance) and HOW the chosen conditions compare to the historical
research data the model was trained on.

HOW TO RUN THIS APP (do this in a terminal, in this folder):
    pip install -r requirements.txt
    streamlit run app.py
It will open automatically in your browser at http://localhost:8501
"""

import streamlit as st          # the web app framework: st.something() draws a UI element
import pandas as pd
import numpy as np
import joblib                   # to load the model files we saved earlier
import matplotlib.pyplot as plt # for the charts


# ---------------------------------------------------------------------------
# 1. PAGE CONFIG — must be the first Streamlit command in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Carbon AI - Biochar Yield Predictor",
    page_icon="🌱",
    layout="wide",              # use the full browser width instead of a narrow centered column
)


# ---------------------------------------------------------------------------
# 2. LOAD THE TRAINED MODEL + SUPPORTING FILES (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource   # tells Streamlit: "run this function once and reuse the result", instead of
                      # reloading the model from disk on every single click, which would be slow
def load_artifacts():
    model = joblib.load("model/biochar_model.pkl")
    encoder = joblib.load("model/feedstock_encoder.pkl")
    medians = joblib.load("model/feature_medians.pkl")
    metrics = joblib.load("model/metrics.pkl")
    importance = pd.read_csv("model/feature_importance.csv", index_col=0)
    history = pd.read_csv("data/clean_biochar_data.csv")   # the training data, used for comparison charts
    return model, encoder, medians, metrics, importance, history

model, feedstock_encoder, medians, metrics, importance_df, history_df = load_artifacts()

FEATURE_COLUMNS = [
    "Feed Stock Type", "Carbon", "Hydrogen", "Nitrogen", "Oxygen",
    "Raw Material Supply (g)", "Temperature (C )", "Residence Time (min)",
    "Gas Flow Rate (L/min)", "Heating Rate (C/min)",
    "Moisture (%)", "VM", "Ash", "FC", "Particle Size (mm)",
]


# ---------------------------------------------------------------------------
# 3. HEADER
# ---------------------------------------------------------------------------
st.title("🌱 Carbon AI — Biochar Yield Predictor")
st.caption(
    "Predicts biochar yield from biomass pyrolysis using a Random Forest model "
    "trained on data compiled from 140 published research papers "
    "(not proprietary data — see References tab for sources)."
)

# st.columns() splits the page horizontally into side-by-side sections.
col_left, col_right = st.columns([1, 2])   # left column is 1/3 width, right column is 2/3 width


# ---------------------------------------------------------------------------
# 4. LEFT COLUMN — INPUT CONTROLS (the "form" the user fills in)
# ---------------------------------------------------------------------------
with col_left:
    st.subheader("Feedstock & Process Inputs")

    # st.selectbox() draws a dropdown menu. sorted(...) makes the list alphabetical.
    feedstock = st.selectbox(
        "Feedstock type",
        sorted(feedstock_encoder.classes_.tolist()),
        index=0,
    )

    # st.slider(label, min, max, default) draws a draggable slider.
    # Default values are set to the median of each column from the training data,
    # so a first-time user sees a "typical" realistic starting point.
    carbon = st.slider("Carbon content (%)", 20.0, 70.0, float(medians["Carbon"]) if "Carbon" not in [] else 45.0)
    hydrogen = st.slider("Hydrogen content (%)", 2.0, 18.0, float(medians["Hydrogen"]))
    nitrogen = st.slider("Nitrogen content (%)", 0.0, 10.0, float(medians["Nitrogen"]))
    oxygen = st.slider("Oxygen content (%)", 5.0, 75.0, float(medians["Oxygen"]))
    moisture = st.slider("Moisture (%)", 0.0, 90.0, float(medians["Moisture (%)"]))
    vm = st.slider("Volatile Matter - VM (%)", 30.0, 95.0, float(medians["VM"]))
    ash = st.slider("Ash content (%)", 0.0, 60.0, float(medians["Ash"]))
    fc = st.slider("Fixed Carbon - FC (%)", 0.0, 35.0, float(medians["FC"]))

    st.markdown("**Reactor process conditions**")
    temperature = st.slider("Pyrolysis Temperature (°C)", 200, 1000, int(medians["Temperature (C )"]))
    residence_time = st.slider("Residence Time (min)", 0, 240, int(medians["Residence Time (min)"]))
    heating_rate = st.slider("Heating Rate (°C/min)", 1, 700, int(medians["Heating Rate (C/min)"]))
    gas_flow = st.slider("Gas Flow Rate (L/min)", 0.0, 11.0, float(medians["Gas Flow Rate (L/min)"]))
    particle_size = st.slider("Particle Size (mm)", 0.0, 12.0, float(medians["Particle Size (mm)"]))
    raw_supply = st.number_input("Raw Material Supply (g)", 0.0, 1000.0, float(medians["Raw Material Supply (g)"]))

    predict_clicked = st.button("🔍 Predict Biochar Yield", use_container_width=True)


# ---------------------------------------------------------------------------
# 5. BUILD A SINGLE-ROW DATAFRAME FROM THE USER'S INPUTS
# ---------------------------------------------------------------------------
# The model expects the SAME column order/names it was trained on, so we
# assemble a one-row DataFrame that mirrors FEATURE_COLUMNS exactly.
input_row = pd.DataFrame([{
    "Feed Stock Type": feedstock_encoder.transform([feedstock])[0],  # convert the chosen name back to its trained integer id
    "Carbon": carbon, "Hydrogen": hydrogen, "Nitrogen": nitrogen, "Oxygen": oxygen,
    "Raw Material Supply (g)": raw_supply,
    "Temperature (C )": temperature, "Residence Time (min)": residence_time,
    "Gas Flow Rate (L/min)": gas_flow, "Heating Rate (C/min)": heating_rate,
    "Moisture (%)": moisture, "VM": vm, "Ash": ash, "FC": fc,
    "Particle Size (mm)": particle_size,
}])[FEATURE_COLUMNS]   # reorder columns to match training order exactly


# ---------------------------------------------------------------------------
# 6. RIGHT COLUMN — PREDICTION + CHARTS
# ---------------------------------------------------------------------------
with col_right:
    st.subheader("Prediction")

    if predict_clicked:
        prediction = model.predict(input_row)[0]   # model.predict() always returns an array; [0] takes the single value

        # st.metric() draws a big highlighted number, good for headline results
        st.metric(
            label=f"Predicted Biochar Yield for {feedstock}",
            value=f"{prediction:.1f} %",
        )
        st.caption(
            f"Model confidence context: on unseen test data this model achieves "
            f"R² = {metrics['r2']:.2f}, average error (MAE) = ±{metrics['mae']:.1f} percentage points."
        )

        # --- Mass balance sanity note -------------------------------------
        st.info(
            "Note: Biochar + Bio-oil + Syngas = 100% of the feedstock mass. "
            "A higher predicted biochar yield generally corresponds to lower "
            "temperature / slower heating (slow pyrolysis regime)."
        )

        # --- Chart 1: where this prediction sits vs historical data --------
        st.markdown("**How this compares to the historical research data**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(history_df["Biochar Yield (%)"], bins=30, color="#8bc34a", alpha=0.7, label="All historical samples")
        ax.axvline(prediction, color="#d32f2f", linewidth=2, label="Your prediction")
        ax.set_xlabel("Biochar Yield (%)")
        ax.set_ylabel("Number of samples")
        ax.legend()
        st.pyplot(fig)

    else:
        st.info("Set your inputs on the left and click **Predict Biochar Yield**.")

    # --- Chart 2: feature importance (always visible) ----------------------
    st.markdown("**What drives biochar yield? (model feature importance)**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    importance_df.sort_values("importance").plot(
        kind="barh", legend=False, ax=ax2, color="#4caf50"
    )
    ax2.set_xlabel("Relative importance")
    st.pyplot(fig2)


# ---------------------------------------------------------------------------
# 7. FOOTER / DATA TRANSPARENCY TAB
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📚 About this data (click to expand)"):
    st.markdown(
        """
        **This is not personal/private data — it is public research data.**
        The dataset compiles pyrolysis experiment results extracted from
        **140 published, peer-reviewed research papers** on biomass pyrolysis
        and biochar production (see `data/SI_Data_References.xlsx` for the
        full list of paper titles, authors, and DOI links).

        The raw data contained formatting inconsistencies from being manually
        compiled across many papers (see `data_preprocessing.py` for exactly
        how these were cleaned).
        """
    )
    st.dataframe(history_df.head(20))
