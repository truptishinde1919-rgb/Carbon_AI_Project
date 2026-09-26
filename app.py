import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Carbon AI - Biochar Yield Predictor",
    page_icon="🌱",
    layout="wide",
)

FEATURE_COLUMNS = [
    "Feed Stock Type", "Carbon", "Hydrogen", "Nitrogen", "Oxygen",
    "Raw Material Supply (g)", "Temperature (C )", "Residence Time (min)",
    "Gas Flow Rate (L/min)", "Heating Rate (C/min)",
    "Moisture (%)", "VM", "Ash", "FC", "Particle Size (mm)",
]


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model/biochar_model.pkl")
        encoder = joblib.load("model/feedstock_encoder.pkl")
        medians = joblib.load("model/feature_medians.pkl")
        metrics = joblib.load("model/metrics.pkl")
        importance = pd.read_csv("model/feature_importance.csv", index_col=0)
        history = pd.read_csv("data/clean_biochar_data.csv")
    except FileNotFoundError as e:
        st.error(
            f"Couldn't find a required file: {e.filename}\n\n"
            "Make sure the `model/` and `data/` folders sit next to app.py."
        )
        st.stop()
    return model, encoder, medians, metrics, importance, history


model, feedstock_encoder, medians, metrics, importance_df, history_df = load_artifacts()

st.title("🌱 Carbon AI — Biochar Yield Predictor")
st.caption(
    "Predicts biochar yield from biomass pyrolysis using a Random Forest model "
    "trained on data compiled from 140 published research papers "
    "(not proprietary data — see the References expander below)."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Feedstock & Process Inputs")

    feedstock_options = sorted(feedstock_encoder.classes_.tolist())
    feedstock = st.selectbox("Feedstock type", feedstock_options, index=0)

    carbon = st.slider("Carbon content (%)", 20.0, 70.0, float(medians["Carbon"]))
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

try:
    feedstock_encoded = feedstock_encoder.transform([feedstock])[0]
except ValueError:
    st.error(f"'{feedstock}' isn't a feedstock type the model was trained on.")
    st.stop()

input_row = pd.DataFrame([{
    "Feed Stock Type": feedstock_encoded,
    "Carbon": carbon, "Hydrogen": hydrogen, "Nitrogen": nitrogen, "Oxygen": oxygen,
    "Raw Material Supply (g)": raw_supply,
    "Temperature (C )": temperature, "Residence Time (min)": residence_time,
    "Gas Flow Rate (L/min)": gas_flow, "Heating Rate (C/min)": heating_rate,
    "Moisture (%)": moisture, "VM": vm, "Ash": ash, "FC": fc,
    "Particle Size (mm)": particle_size,
}])[FEATURE_COLUMNS]

with col_right:
    st.subheader("Prediction")

    if predict_clicked:
        try:
            prediction = model.predict(input_row)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.metric(
            label=f"Predicted Biochar Yield for {feedstock}",
            value=f"{prediction:.1f} %",
        )
        st.caption(
            f"Model confidence context: on unseen test data this model achieves "
            f"R² = {metrics['r2']:.2f}, average error (MAE) = ±{metrics['mae']:.1f} percentage points."
        )

        st.info(
            "Biochar + Bio-oil + Syngas = 100% of the feedstock mass. "
            "A higher predicted biochar yield generally corresponds to lower "
            "temperature / slower heating (slow pyrolysis regime)."
        )

        st.markdown("**How this compares to the historical research data**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(history_df["Biochar Yield (%)"], bins=30, color="#8bc34a", alpha=0.7, label="All historical samples")
        ax.axvline(prediction, color="#d32f2f", linewidth=2, label="Your prediction")
        ax.set_xlabel("Biochar Yield (%)")
        ax.set_ylabel("Number of samples")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Set your inputs on the left and click **Predict Biochar Yield**.")

    st.markdown("**What drives biochar yield? (model feature importance)**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    importance_df.sort_values("importance").plot(kind="barh", legend=False, ax=ax2, color="#4caf50")
    ax2.set_xlabel("Relative importance")
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()
with st.expander("📚 About this data"):
    st.markdown(
        """
        This is public research data, not proprietary or private data.
        It's compiled from **140 published, peer-reviewed papers** on
        biomass pyrolysis and biochar production — see
        `data/SI_Data_References.xlsx` for titles, authors, and DOIs.

        The raw data had formatting inconsistencies from being manually
        pulled from many papers; see `data_preprocessing.py` for how it
        was cleaned.
        """
    )
    st.dataframe(history_df.head(20))