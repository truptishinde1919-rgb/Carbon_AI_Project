# 🌱 Carbon AI — Biochar Yield Predictor

**Carbon AI** is a machine-learning based application that predicts **biochar yield (%)** from biomass pyrolysis conditions and feedstock properties.

The project uses a **Random Forest Regression** model trained on publicly available experimental research data compiled from **140 published research papers** on biomass pyrolysis.

> **Data source:** Publicly available research data. No personal or proprietary data is used.

---

## 🎯 Project Objective

The main objective of Carbon AI is to provide a simple data-driven tool for estimating biochar yield based on different biomass and pyrolysis parameters.

The application allows users to enter experimental conditions such as:

* Feedstock type
* Carbon, Hydrogen, Nitrogen and Oxygen content
* Raw material supply
* Pyrolysis temperature
* Residence time
* Gas flow rate
* Heating rate
* Moisture
* Volatile matter (VM)
* Ash content
* Fixed carbon (FC)
* Particle size

The trained machine-learning model then predicts the expected **Biochar Yield (%)**.

---

## 🧠 Machine Learning Approach

The project follows three main stages:

```text
Raw Research Dataset
        ↓
Data Preprocessing & Cleaning
        ↓
Feature Preparation
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Streamlit Dashboard
        ↓
Biochar Yield Prediction
```

### Model Used

**RandomForestRegressor**

The model combines multiple decision trees and averages their predictions to estimate biochar yield.

---

## 📊 Dataset

The dataset is based on experimental biomass pyrolysis research data.

### Dataset Information

| Information        | Details                           |
| ------------------ | --------------------------------- |
| Total records      | 1,674 rows                        |
| Biomass feedstocks | 70                                |
| Research papers    | 140                               |
| Target variable    | Biochar Yield (%)                 |
| Data type          | Public research/experimental data |
| Raw dataset        | `SI_Data.xlsx`                    |
| Reference file     | `SI_Data_References.xlsx`         |

The `SI_Data_References.xlsx` file contains references associated with the research data sources.

---

## 🧹 Data Preprocessing

The raw Excel dataset required cleaning because the data was compiled from multiple published research sources and contained formatting inconsistencies.

The preprocessing stage includes:

* Cleaning numerical columns
* Handling incorrectly formatted values
* Removing unusable target values
* Handling missing values
* Converting relevant columns into numerical format
* Encoding categorical feedstock information
* Creating a clean dataset for model training

Examples of issues handled during preprocessing included incorrectly formatted values in:

* Oxygen
* Gas Flow Rate
* Heating Rate
* Biochar Yield

The cleaned dataset is saved as:

```text
data/clean_biochar_data.csv
```

---

## 🤖 Model Training

The cleaned dataset is used to train a **Random Forest Regression** model.

### Model Configuration

```text
Algorithm: RandomForestRegressor
Number of trees: 300
Target: Biochar Yield (%)
```

The trained model and supporting files are stored inside the `model/` directory.

---

## 📈 Model Performance

The model was evaluated on held-out test data.

| Metric   |                   Result |
| -------- | -----------------------: |
| R² Score |                  ≈ 0.817 |
| MAE      | ≈ 2.60 percentage points |
| RMSE     | ≈ 4.88 percentage points |

### What these metrics mean

**R² ≈ 0.817**

The model explains approximately 81.7% of the variation in biochar yield in the held-out test dataset.

**MAE ≈ 2.60**

On average, the prediction differs from the actual biochar yield by approximately **2.6 percentage points**.

**RMSE ≈ 4.88**

This measures prediction error while giving greater weight to larger errors.

> These results are evaluation results on the project's test split and should not be interpreted as guaranteed performance on new experimental conditions outside the training data.

---

## 🔍 Important Features

Feature-importance analysis from the trained Random Forest indicates that several feedstock and pyrolysis parameters contribute to prediction.

Important variables include:

* **Temperature**
* **Ash content**
* **Carbon content**
* Residence time
* Moisture
* Fixed carbon
* Heating rate
* Feedstock characteristics

The project stores feature-importance information in:

```text
model/feature_importance.csv
```

---

## 🖥️ Streamlit Dashboard

Carbon AI includes an interactive **Streamlit dashboard**.

The dashboard allows users to:

1. Select a biomass feedstock.
2. Enter feedstock properties.
3. Enter pyrolysis conditions.
4. Submit the input values.
5. Generate a predicted biochar yield.
6. View relevant model information and prediction history.

The application is designed to make the machine-learning model easier to use without requiring users to write Python code.

---

## 📁 Project Structure

```text
CarbonAI_Project/
│
├── data/
│   ├── SI_Data.xlsx
│   ├── SI_Data_References.xlsx
│   └── clean_biochar_data.csv
│
├── model/
│   ├── biochar_model.pkl
│   ├── dashboard_config.json
│   ├── feature_importance.csv
│   ├── feature_medians.pkl
│   ├── feedstock_encoder.pkl
│   └── metrics.pkl
│
├── data_preprocessing.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── start_carbon_ai.bat
```

---

## ⚙️ Technologies Used

### Programming & Data Science

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

### Machine Learning

* Random Forest Regression
* Feature preprocessing
* Missing-value handling
* Categorical encoding
* Model evaluation
* Feature importance analysis

### Dashboard

* Streamlit

### Data Source

* Published biomass pyrolysis research data

---

## 🚀 How to Run the Project

### 1. Clone or download the project

Open the project folder in VS Code or a terminal.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess the dataset

```bash
python data_preprocessing.py
```

This creates:

```text
data/clean_biochar_data.csv
```

### 4. Train the model

```bash
python train_model.py
```

This creates the model artifacts inside:

```text
model/
```

### 5. Start the dashboard

```bash
streamlit run app.py
```

The dashboard will normally open at:

```text
http://localhost:8501
```

---

## 🌐 Live Application

The Carbon AI dashboard is also deployed online using Streamlit.

**Live Demo:**

https://carbonaiproject-rpwqui2btxgdzq5qsctswz.streamlit.app/

---

## 💡 Why Carbon AI?

Biochar production depends on several factors, including:

* Biomass/feedstock properties
* Pyrolysis temperature
* Residence time
* Heating rate
* Moisture
* Gas flow
* Particle size

Testing every combination experimentally can require significant time and resources.

Carbon AI demonstrates how machine learning can be used to estimate biochar yield from these experimental parameters and provide a quick prediction through an interactive dashboard.

---

## ⚠️ Limitations

Carbon AI is a machine-learning research/educational prototype.

Predictions depend on the quality and range of the published data used for training.

The model should therefore be used as a **prediction and research-support tool**, not as a replacement for laboratory experiments.

Predictions for conditions substantially outside the training data may be less reliable.

---

## 🔮 Future Improvements

Possible future improvements include:

* Adding more experimental datasets
* Increasing the number of biomass feedstocks
* Comparing multiple ML algorithms
* Hyperparameter optimization
* Cross-validation
* Explainable AI techniques such as SHAP
* Prediction confidence/uncertainty estimates
* Additional visualization and analytics
* Integration of more published research data
* Improved model generalization to new feedstocks
* Downloadable prediction reports

---

## 👥 Project Team

**Carbon AI — Biochar Yield Predictor**

Machine Learning + Data Science + Climate/Biochar Research Project

---

## 📚 Data References

The research sources used to compile the dataset are provided in:

```text
data/SI_Data_References.xlsx
```

The project uses publicly available research data compiled from published biomass pyrolysis studies.

---

## 📌 Project Summary

**Carbon AI is a machine-learning application that predicts biochar yield from biomass pyrolysis conditions and feedstock properties. A Random Forest Regression model is trained on 1,674 experimental records compiled from 140 published research papers. The project combines data preprocessing, machine learning, model evaluation, feature-importance analysis, and an interactive Streamlit dashboard into one end-to-end application.**

