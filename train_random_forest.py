import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay


# Load dataset
df = pd.read_csv("dataset/heart_2020_cleaned.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())


# Convert target column to numbers
df["HeartDisease"] = df["HeartDisease"].map({
    "Yes": 1,
    "No": 0
})


# Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]


# Numerical columns
numeric_columns = [
    "BMI",
    "PhysicalHealth",
    "MentalHealth",
    "SleepTime"
]


# Categorical columns
categorical_columns = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer"
]


# Split dataset into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Preprocess numerical columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# Preprocess categorical columns
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_columns),
    ("cat", categorical_transformer, categorical_columns)
])


# Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", random_forest_model)
])


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Evaluate model
print("\nRandom Forest Results")
print("---------------------")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Visualised confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Heart Disease", "Heart Disease"],
    cmap="Greens",
    values_format="d"
)

plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()


plt.savefig("results/random_forest_confusion_matrix.png", dpi=300)

plt.show()


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_prob))


# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully in models/random_forest_model.pkl")