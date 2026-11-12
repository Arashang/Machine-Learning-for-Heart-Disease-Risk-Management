# Heart Disease Risk Prediction Web App

A web application that estimates a person's risk of heart disease using machine learning, and gives personalised, easy-to-understand recommendations.

It is designed for **ordinary users, not clinicians**. Instead of relying on clinical measurements such as blood tests or ECG readings, the model uses only information people can report about themselves — such as age, lifestyle habits and general health — so anyone can check their risk without a medical appointment.

> ⚠️ **Disclaimer:** This tool is for educational and awareness purposes only. It does not provide a medical diagnosis and is not a substitute for professional medical advice. If you have concerns about your heart health, please speak to a healthcare professional.

## Features

- **Risk prediction** from self-reported health and lifestyle information
- **Three machine learning models** trained and compared: Logistic Regression, Random Forest and XGBoost (Logistic Regression is used in the app)
- **Probability-based risk levels:** Low (under 10%), Moderate (10–50%) and High (50% and above)
- **Rule-based recommendations** linked to the user's identified risk factors (e.g. smoking, low physical activity)
- **AI-enhanced explanations (optional):** Google Gemini (2.5 Flash) rewrites the rule-based recommendations into a clear, prioritised action plan. It is instructed not to diagnose, not to invent new medical advice, and to use only the information the app provides. If no API key is set or the API is unavailable, the app falls back to the standard rule-based text, so it works fully without one.
- **Web interface** built with Flask for entering details and viewing results

**User inputs:** height and weight (used for BMI), smoking, heavy alcohol use, stroke history, physical and mental health days, difficulty walking, sex, age group, ethnicity, diabetes, physical activity, general health, sleep hours, asthma, kidney disease and skin cancer.

## How it works

```
User inputs → ML model (Logistic Regression) → probability & risk level
            → rule-based recommendations → Gemini explanation (or fallback text) → results page
```

Gemini is used only as a communication layer. The prediction always comes from the trained model, and the recommendations always come from the app's own rules.

## Design priority: catching at-risk people

In health screening, missing someone who is at risk is far worse than a false alarm. Because the dataset is heavily imbalanced (far more "no heart disease" records), the models use class weighting (`class_weight="balanced"` for scikit-learn models, `scale_pos_weight` for XGBoost) so they pay more attention to the at-risk class. This favours **recall (sensitivity) over precision**: the models accept more false positives so that fewer genuine at-risk cases are missed. A high-risk result is meant as a prompt to see a doctor, not a diagnosis.

## Built with

- **Python**
- **pandas** – data loading and preparation
- **scikit-learn** – preprocessing (`SimpleImputer`, `StandardScaler`, train-test split), Logistic Regression, Random Forest, evaluation
- **XGBoost** – gradient-boosted model
- **Flask** – web application
- **Google Gemini API** – optional AI-written recommendations
- **HTML / CSS** – web interface (`templates/`, `static/`)

## Results

All models were evaluated on a held-out, stratified test set of **63,959 records** (20% of the data; 5,475 with heart disease). Metrics for precision, recall and F1-score refer to the heart disease class.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (used in app) | 0.75 | 0.22 | 0.77 | 0.35 | **0.8372** |
| Random Forest | 0.75 | 0.22 | 0.77 | 0.35 | 0.8315 |
| XGBoost | 0.75 | 0.22 | 0.77 | 0.34 | 0.8368 |

**Logistic Regression confusion matrix (test set):**

| | Predicted: No Heart Disease | Predicted: Heart Disease |
|---|---|---|
| **Actual: No Heart Disease** | 43,661 | 14,823 |
| **Actual: Heart Disease** | 1,233 | 4,242 |

**Key findings:**
- All three models performed almost identically, which suggests performance was limited mainly by the available self-reported features rather than the choice of algorithm.
- **Logistic Regression was chosen for the app** because it achieved the highest ROC-AUC while being the simplest and most interpretable model.
- The models detect about **77% of actual heart disease cases**, at the cost of low precision (many false alarms). This is an intentional trade-off for a risk-awareness tool.

## Limitations

- **Not a diagnostic tool.** The models are trained on survey data and cannot replace clinical assessment.
- **Low precision.** Many users without heart disease are flagged as higher risk.
- **Missing clinical features.** The dataset does not include blood pressure, cholesterol, family history or resting heart rate.
- **Self-reported data** may contain recall bias or inaccurate answers.
- **Generative AI risk.** Despite strict prompt rules, LLM output can still contain inaccuracies, which is why the app always shows a disclaimer.

## Future improvements

- Optional clinical inputs (e.g. blood pressure, cholesterol, family history) for users who know them
- Integration of wearable-device data (heart rate, activity, sleep) for continuous risk monitoring
- More rigorous validation of the models

## Project structure

```
├── app.py                          # Main Flask web app – run this
├── recommendation_system.py        # Rule-based recommendations from risk factors
├── gemini_helper.py                # Optional Gemini rewrite of recommendations
├── train_logistic_regression.py    # Trains the Logistic Regression model
├── train_random_forest.py          # Trains the Random Forest model
├── train_xgboost.py                # Trains the XGBoost model
├── models/                         # Saved trained models and preprocessing objects
├── results/                        # Evaluation results
├── dataset/                        # Training data
├── templates/                      # HTML pages
├── static/                         # CSS, images and other static files
└── requirements.txt                # Python dependencies
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/cardiovascular-risk-prediction.git
cd cardiovascular-risk-prediction
```

### 2. Create a virtual environment and install requirements

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Add a Gemini API key

To enable AI-written recommendations, get a free API key from [Google AI Studio](https://aistudio.google.com/) and set it as an environment variable:

macOS / Linux:

```bash
export GEMINI_API_KEY="your-key-here"
```

Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

Never commit your API key to the repository. Without a key, the app still works using rule-based recommendations.

### 4. Run the app

```bash
python app.py
```

Then open the local address shown in the terminal in your browser.

### Retraining the models (optional)

```bash
python train_logistic_regression.py
python train_random_forest.py
python train_xgboost.py
```

## Data

This project uses the 2020 **Behavioral Risk Factor Surveillance System (BRFSS)** Heart Disease dataset — a cleaned version of the annual health survey run by the Centers for Disease Control and Prevention (CDC), released under CC0 1.0 (public domain).

Source: Centers for Disease Control and Prevention (CDC). Behavioral Risk Factor Surveillance System Survey Data, 2020. https://www.cdc.gov/brfss/annual_data/annual_2020.html

## About

Developed by **Arash Ahangar** as a final-year project (CST3990 Undergraduate Individual Project) for a BSc in Computer Science at Middlesex University (2026).

## License

The code in this repository is licensed under the [MIT License](LICENSE).
The dataset is in the public domain (CC0 1.0); see the Data section above for the source.
