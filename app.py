import joblib
import pandas as pd


from flask import Flask, render_template, request

from recommendation_system import generate_rule_based_recommendations
from gemini_helper import improve_recommendations_with_gemini


app = Flask(__name__)


# Load saved Logistic Regression model and preprocessors

model = joblib.load("models/brfss_logistic_regression_model.pkl")

scaler = joblib.load("models/scaler.pkl")
num_imputer = joblib.load("models/num_imputer.pkl")
cat_imputer = joblib.load("models/cat_imputer.pkl")

model_columns = joblib.load("models/model_columns.pkl")
numeric_columns = joblib.load("models/numeric_columns.pkl")
categorical_columns = joblib.load("models/categorical_columns.pkl")


# Prepare user input before sending it to the ML model

def prepare_user_input(user_input):
    
    #preprocess user input to match the format used during model training

    user_input = user_input.copy()

    user_input[numeric_columns] = num_imputer.transform(user_input[numeric_columns])

    user_input[categorical_columns] = cat_imputer.transform(user_input[categorical_columns])

    user_input = pd.get_dummies(
        user_input,
        columns=categorical_columns
    )

    user_input = user_input.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale numerical columns
    user_input[numeric_columns] = scaler.transform(user_input[numeric_columns])

    return user_input


# BMI calculation

def calculate_bmi(height_cm, weight_kg):

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    return round(bmi, 2)


# Recommendation system

def generate_recommendation(user_data, probability):


# combine rule-based recommendations with LLM improvements

    recommendation_result = generate_rule_based_recommendations(
        user_data=user_data,
        probability=probability
    )

    risk_level = recommendation_result["risk_level"]
    risk_factors = recommendation_result["risk_factors"]
    recommendations = recommendation_result["recommendations"]

    llm_recommendation = improve_recommendations_with_gemini(
        risk_level=risk_level,
        probability=probability,
        risk_factors=risk_factors,
        recommendations=recommendations
    )

    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "llm_recommendation": llm_recommendation
    }


# Home page

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        probability=None,
        risk_level=None,
        risk_factors=None,
        recommendations=None,
        llm_recommendation=None,
        bmi=None,
        error=None
    )


# Prediction route


@app.route("/predict", methods=["POST"])
def predict():
    try:
    
        # Collect the user input from the form

        height = float(request.form.get("Height"))
        weight = float(request.form.get("Weight"))

        bmi = calculate_bmi(height, weight)
        smoking = request.form.get("Smoking")
        alcohol_drinking = request.form.get("AlcoholDrinking")
        stroke = request.form.get("Stroke")
        physical_health = float(request.form.get("PhysicalHealth"))
        mental_health = float(request.form.get("MentalHealth"))
        diff_walking = request.form.get("DiffWalking")
        sex = request.form.get("Sex")
        age_category = request.form.get("AgeCategory")
        race = request.form.get("Race")
        diabetic = request.form.get("Diabetic")
        physical_activity = request.form.get("PhysicalActivity")
        gen_health = request.form.get("GenHealth")
        sleep_time = float(request.form.get("SleepTime"))
        asthma = request.form.get("Asthma")
        kidney_disease = request.form.get("KidneyDisease")
        skin_cancer = request.form.get("SkinCancer")

        # Create DataFrame for the ML model

        user_input = pd.DataFrame([{
            "BMI": bmi,
            "Smoking": smoking,
            "AlcoholDrinking": alcohol_drinking,
            "Stroke": stroke,
            "PhysicalHealth": physical_health,
            "MentalHealth": mental_health,
            "DiffWalking": diff_walking,
            "Sex": sex,
            "AgeCategory": age_category,
            "Race": race,
            "Diabetic": diabetic,
            "PhysicalActivity": physical_activity,
            "GenHealth": gen_health,
            "SleepTime": sleep_time,
            "Asthma": asthma,
            "KidneyDisease": kidney_disease,
            "SkinCancer": skin_cancer
        }])

        # Create dictionary version for recommendations

        user_data = {
            "Height": height,
            "Weight": weight,
            "BMI": bmi,
            "Smoking": smoking,
            "AlcoholDrinking": alcohol_drinking,
            "Stroke": stroke,
            "PhysicalHealth": physical_health,
            "MentalHealth": mental_health,
            "DiffWalking": diff_walking,
            "Sex": sex,
            "AgeCategory": age_category,
            "Race": race,
            "Diabetic": diabetic,
            "PhysicalActivity": physical_activity,
            "GenHealth": gen_health,
            "SleepTime": sleep_time,
            "Asthma": asthma,
            "KidneyDisease": kidney_disease,
            "SkinCancer": skin_cancer
        }

        # Preprocess user input

        processed_input = prepare_user_input(user_input)

        # Make prediction

        probability_value = model.predict_proba(processed_input)[0][1]
        probability_percentage = round(probability_value * 100, 2)

        if probability_percentage >= 50:
           prediction_text = "The result shows a high risk pattern."
        elif probability_percentage >= 10:
             prediction_text = "The result shows a moderate risk pattern."
        else:
             prediction_text = "The result shows a low risk pattern"
        # Generate recommendations
        
        recommendation_output = generate_recommendation(
            user_data=user_data,
            probability=probability_percentage
        )

        # Send result back to page

        return render_template(
            "index.html",
            prediction=prediction_text,
            probability=probability_percentage,
            risk_level=recommendation_output["risk_level"],
            risk_factors=recommendation_output["risk_factors"],
            recommendations=recommendation_output["recommendations"],
            llm_recommendation=recommendation_output["llm_recommendation"],
            bmi=bmi,
            error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            risk_level=None,
            risk_factors=None,
            recommendations=None,
            llm_recommendation=None,
            bmi=None,
            error=f"An error occurred: {str(e)}"
        )


# Run Flask app

if __name__ == "__main__":
    app.run(debug=False, port=5000)