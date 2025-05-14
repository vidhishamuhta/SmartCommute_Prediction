# Model Training and Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))

# Hyperparameter Optimization for Random Forest
rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}
gs_rf = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs_rf.fit(X_train, y_train)

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

output_dir = 'C:/Users/Vidhisha/Documents/models/'  # Replace with a path you have access to
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Best Model
best_rf = gs_rf.best_estimator_
joblib.dump(best_rf, 'C:/Users/Vidhisha/Documents/models/best_random_forest.pkl')

# Feature Importance
feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features - Random Forest")
plt.show()

