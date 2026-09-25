# Carbon AI — Biochar Yield Predictor

Predicts **biochar yield (%)** from biomass pyrolysis using a Random Forest model,
trained on data compiled from 140 published research papers on biomass pyrolysis
(see `data/SI_Data_References.xlsx`). **This is public research data, not personal
or proprietary data.**

## Project structure
```
CarbonAI_Project/
├── data/
│   ├── SI_Data.xlsx                 # raw dataset (as provided)
│   ├── SI_Data_References.xlsx      # citations for every data source
│   └── clean_biochar_data.csv       # created after running data_preprocessing.py
├── model/                            # created after running train_model.py
│   ├── biochar_model.pkl
│   ├── feedstock_encoder.pkl
│   ├── feature_medians.pkl
│   ├── metrics.pkl
│   └── feature_importance.csv
├── data_preprocessing.py             # Step 1: cleans the raw data
├── train_model.py                    # Step 2: trains + evaluates the model
├── app.py                            # Step 3: the Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to run (3 commands)
```bash
pip install -r requirements.txt
python data_preprocessing.py
python train_model.py
streamlit run app.py
```
The dashboard opens automatically in your browser at `http://localhost:8501`.

## Model performance
- R² = 0.82 on held-out test data
- MAE ≈ 2.6 percentage points
- Biggest drivers of yield: **Temperature**, **Ash content**, **Carbon content**

## Notes for your project report / viva
- Dataset: 1,674 rows, 70 biomass feedstocks, compiled from 140 papers.
- Target variable: Biochar Yield (%).
- Data cleaning was needed because the raw Excel file had formatting corruption
  from being manually compiled across many source PDFs (see comments in
  `data_preprocessing.py` for exact examples and fixes).
- Missing values were imputed using column medians.
- Model: RandomForestRegressor (scikit-learn), 300 trees.
