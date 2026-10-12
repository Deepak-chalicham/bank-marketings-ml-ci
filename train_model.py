import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


print("Loading Bank Marketing dataset...")

data = pd.read_csv(
    "bank-additional-full.csv",
    sep=";"
)

print("Dataset loaded successfully.")
print("Number of records:", len(data))
print("Number of columns:", len(data.columns))


data["y"] = data["y"].map({
    "yes": 1,
    "no": 0
})


X = data.drop("y", axis=1)
y = data["y"]


numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions
)


print("\nModel Evaluation")
print("----------------")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix:")
print(matrix)


joblib.dump(
    model,
    "bank_marketing_model.pkl"
)

print("\nModel saved as bank_marketing_model.pkl")


metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}


with open(
    "metrics.json",
    "w"
) as file:
    json.dump(
        metrics,
        file,
        indent=4
    )

print("Metrics saved as metrics.json")