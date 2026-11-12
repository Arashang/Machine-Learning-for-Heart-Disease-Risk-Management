
#function to generate rule-based recommendations based on user data and predicted probability

def generate_rule_based_recommendations(user_data, probability):


    recommendations = []
    risk_factors = []

    # Risk level based on percentage probability

    if probability < 10:
        risk_level = "Low"
    elif probability < 50:
        risk_level = "Moderate"
    else:
        risk_level = "High"



    # Smoking

    if user_data.get("Smoking") == "Yes":
        risk_factors.append("Smoking")
        recommendations.append(
            "Smoking is a major modifiable cardiovascular risk factor. Stopping smoking can significantly reduce heart disease risk."
        )

    # Physical activity

    if user_data.get("PhysicalActivity") == "No":
        risk_factors.append("Low physical activity")
        recommendations.append(
            "Low physical activity may increase cardiovascular risk. Regular moderate exercise may improve heart health."
        )

    # Alcohol

    if user_data.get("AlcoholDrinking") == "Yes":
        risk_factors.append("Heavy alcohol drinking")
        recommendations.append(
            "Heavy alcohol consumption may increase cardiovascular risk. Reducing alcohol intake may support better heart health."
        )

    # Stroke

    if user_data.get("Stroke") == "Yes":
        risk_factors.append("Previous stroke")
        recommendations.append(
            "A previous stroke is an important health risk factor. You should discuss cardiovascular prevention with a healthcare professional."
        )

    # Difficulty walking

    if user_data.get("DiffWalking") == "Yes":
        risk_factors.append("Difficulty walking")
        recommendations.append(
            "Difficulty walking may reflect reduced mobility or poorer general health. Gentle physical activity, if safe, may be beneficial."
        )

    # Diabetes

    diabetic_status = user_data.get("Diabetic", "")

    if diabetic_status in ["Yes", "Yes (during pregnancy)"]:
        risk_factors.append("Diabetes")
        recommendations.append(
            "Diabetes is associated with increased cardiovascular risk. Blood sugar control and regular medical check-ups are important."
        )
    elif diabetic_status == "No, borderline diabetes":
        risk_factors.append("Borderline diabetes")
        recommendations.append(
            "Borderline diabetes may increase future cardiovascular risk. Lifestyle changes and regular monitoring may help."
        )

    # General health

    if user_data.get("GenHealth") in ["Fair", "Poor"]:
        risk_factors.append("Poor general health")
        recommendations.append(
            "Your general health rating suggests that a wider health review may be useful."
        )

    # Sleep

    sleep_time = float(user_data.get("SleepTime", 0))

    if sleep_time < 6:
        risk_factors.append("Low sleep duration")
        recommendations.append(
            "Short sleep duration may negatively affect overall health. Improving sleep habits may support cardiovascular wellbeing."
        )
    elif sleep_time > 9:
        risk_factors.append("Long sleep duration")
        recommendations.append(
            "Very long sleep duration can sometimes be linked with underlying health issues. Consider discussing this with a healthcare professional."
        )

    # Kidney disease

    if user_data.get("KidneyDisease") == "Yes":
        risk_factors.append("Kidney disease")
        recommendations.append(
            "Kidney disease is linked with increased cardiovascular risk. Regular clinical monitoring is important."
        )

    # Asthma

    if user_data.get("Asthma") == "Yes":
        risk_factors.append("Asthma")
        recommendations.append(
            "Asthma may affect general health and physical activity. Good asthma management may support overall wellbeing."
        )


    # Default recommendation

    if not recommendations:
        recommendations.append(
            "No major modifiable risk factors were identified from the entered information. Maintaining a healthy lifestyle is still important."
        )

    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendations": recommendations
    }