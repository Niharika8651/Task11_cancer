# Task 11: SVM Breast Cancer Classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:\n", y.value_counts())

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Pipeline (Scaler + SVM)
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

# -----------------------------
# 4. Hyperparameter Tuning
# -----------------------------
param_grid = {
    "svm__kernel": ["rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": [0.01, 0.1, 1]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# -----------------------------
# 6. Evaluation
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 7. ROC Curve & AUC
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM Breast Cancer")
plt.legend()
plt.show()

print("ROC AUC Score:", roc_auc)

# -----------------------------
# 8. Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/svm_model.pkl")

print("\nModel saved successfully at model/svm_model.pkl")
