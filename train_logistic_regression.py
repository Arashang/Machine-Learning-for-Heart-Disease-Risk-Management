import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


# Load and check the dataset
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


# numerical columns
numeric_columns = [
    "BMI",
    "PhysicalHealth",
    "MentalHealth",
    "SleepTime"
]


# categorical columns
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


print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Handle missing values in numerical columns
num_imputer = SimpleImputer(strategy="median")

num_imputer.fit(X_train[numeric_columns])

X_train[numeric_columns] = num_imputer.transform(X_train[numeric_columns])
X_test[numeric_columns] = num_imputer.transform(X_test[numeric_columns])


# Handle missing values in categorical columns
cat_imputer = SimpleImputer(strategy="most_frequent")

cat_imputer.fit(X_train[categorical_columns])

X_train[categorical_columns] = cat_imputer.transform(X_train[categorical_columns])
X_test[categorical_columns] = cat_imputer.transform(X_test[categorical_columns])


# Convert categorical columns into dummy variables
X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)


# Make sure test set has the same columns as training set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


print("\nTraining data after conversion:", X_train.shape)
print("Testing data after conversion:", X_test.shape)


# Scale numerical columns
scaler = StandardScaler()

scaler.fit(X_train[numeric_columns])

X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])


# Create logistic regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# Train the model
print("\nTraining the Logistic Regression model...")
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Evaluate the model
print("\n=== Logistic Regression Results ===")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", round(auc_score, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Visualised confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Heart Disease", "Heart Disease"],
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig("results/logistic_regression_confusion_matrix.png", dpi=300)

plt.show()


# Save model and preprocessing objects
joblib.dump(model, "models/brfss_logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(num_imputer, "models/num_imputer.pkl")
joblib.dump(cat_imputer, "models/cat_imputer.pkl")
joblib.dump(X_train.columns.tolist(), "models/model_columns.pkl")
joblib.dump(numeric_columns, "models/numeric_columns.pkl")
joblib.dump(categorical_columns, "models/categorical_columns.pkl")

print("\nModel and preprocessing files saved successfully.")