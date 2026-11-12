import os
from google import genai

# function to improve recommendations using Gemini

def improve_recommendations_with_gemini(risk_level, probability, risk_factors, recommendations):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback_recommendation_text(risk_level, probability, risk_factors, recommendations)

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
        You are helping rewrite cardiovascular risk recommendations for a web app.

        Important rules:
        - Do not diagnose the user.
        - Do not invent new medical advice.
        - Only use the provided risk factors and recommendations.
        - Keep the language simple and understandable.
        - Include a clear disclaimer that this is not medical diagnosis.
        - Give the user a clear, actionable plan based only on the provided recommendations, prioritising the most serious or urgent risk factors first.
        - Recommend speaking to a healthcare professional, especially for high risk.

        Predicted risk level: {risk_level}
        Estimated probability: {probability}%

        Risk factors identified:
        {risk_factors}

        Rule-based recommendations:
        {recommendations}

        Write a short personalised recommendation in 2 to 4 paragraphs.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return fallback_recommendation_text(risk_level, probability, risk_factors, recommendations)

    except Exception as error:
        print("Gemini error:", error)
        return fallback_recommendation_text(risk_level, probability, risk_factors, recommendations)


# This function is used when Gemini is not available, to ensure the user still gets a recommendation.

def fallback_recommendation_text(risk_level, probability, risk_factors, recommendations):

    text = f"Your predicted cardiovascular risk level is {risk_level}, with an estimated probability of {probability}%.\n\n"

    if risk_factors:
        text += "The main risk factors identified from your answers are: "
        text += ", ".join(risk_factors)
        text += ".\n\n"

    text += "Recommendations:\n"

    for item in recommendations:
        text += f"- {item}\n"

    return text