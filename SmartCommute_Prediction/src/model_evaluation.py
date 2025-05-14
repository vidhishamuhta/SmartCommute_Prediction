import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('../data/commute_data.csv')

# Preprocess: (make sure this matches training)
# NOTE: Reuse your final features pipeline or preprocessing code here
# Here's an example assuming it was done before:
from src.model_training import preprocess_data
X, y = preprocess_data(data)

# Load model
model = joblib.load('../models/best_random_forest.pkl')

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Evaluation
print("Classification Report:")
print(classification_report(y, y_pred))

print("ROC AUC Score:", roc_auc_score(y, y_proba))

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
