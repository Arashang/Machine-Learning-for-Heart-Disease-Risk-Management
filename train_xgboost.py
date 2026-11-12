import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("dataset/heart_2020_cleaned.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())


# Convert target column into numbers
df["HeartDisease"] = df["HeartDisease"].map({
    "Yes": 1,
    "No": 0
})


# Separate input features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]


# Convert categorical columns into dummy columns
X = pd.get_dummies(X, drop_first=True)

print("\nShape after encoding:", X.shape)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Handle class imbalance
negative_cases = (y_train == 0).sum()
positive_cases = (y_train == 1).sum()

scale_pos_weight = negative_cases / positive_cases

print("\nNegative cases:", negative_cases)
print("Positive cases:", positive_cases)
print("Scale pos weight:", scale_pos_weight)


# Create XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Visualised confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Heart Disease", "Heart Disease"],
    cmap="Oranges",
    values_format="d"
)

plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()

plt.savefig("results/xgboost_confusion_matrix.png", dpi=300)

plt.show()


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_prob))


# Save model and columns

joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(X.columns.tolist(), "models/xgboost_columns.pkl")

print("\nXGBoost model saved successfully.")
print("Saved as: models/xgboost_model.pkl")
print("Saved columns as: models/xgboost_columns.pkl")