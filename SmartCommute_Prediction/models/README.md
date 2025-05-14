
## 🧠 Objective

Predict if a commuter will be delayed by **more than 10 minutes**, allowing the app to notify users in advance.

## 📊 Features Used

- **Distance (in km)** using Haversine formula
- **Temporal features** from `start_time` (hour, day of week)
- **Categorical encodings** for mode of transport, weather, etc.
- **Traffic density**, public holidays, road work indicators
- **Historical delay**

## 🔍 EDA & Feature Engineering

- Visualized delays, traffic patterns, and transport modes
- Handled missing values and categorical features
- Engineered `distance_km` using geolocation
- Created binary target variable: `is_delayed > 10 mins`

## 🧪 Model Development

- Models trained: `Logistic Regression`, `Random Forest`, `XGBoost`
- Used `GridSearchCV` for hyperparameter optimization
- Feature importance analysis performed
- Best model saved using `joblib`

## 📈 Model Evaluation & Error Analysis

- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Error Analysis on False Positives & Negatives
- Identified transport mode and time-of-day bias
- Suggestions: Use route-specific delay histories for better accuracy

## ⚙️ Tech Stack

- Python 3.10
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn, XGBoost
- Joblib, Jupyter Notebook

## 🚀 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/SmartCommute-Prediction.git
cd SmartCommute-Prediction

# Install dependencies
pip install -r requirements.txt

# Run training
python src/model_training.py

# Run evaluation
python src/model_evaluation.py
