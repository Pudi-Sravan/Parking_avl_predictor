import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('info.csv')

# Clean and parse timestamps
df['checkin_timestamp'] = pd.to_datetime(df['checkin_timestamp'].astype(str).str.strip(), errors='coerce')
df['checkout_timestamp'] = pd.to_datetime(df['checkout_timestamp'].astype(str).str.strip(), errors='coerce')

print("Checkin parsing errors:", df['checkin_timestamp'].isna().sum())
print("Checkout parsing errors:", df['checkout_timestamp'].isna().sum())

# Extract hour of day
df['hour_of_day'] = df['checkin_timestamp'].dt.hour

# Encode categorical features
le_day = LabelEncoder()
df['day_of_week_enc'] = le_day.fit_transform(df['day_of_week'])

le_slot = LabelEncoder()
df['slot_type_enc'] = le_slot.fit_transform(df['slot_type'])

# Convert event day to binary
df['is_event_day'] = df['is_event_day'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

# Drop rows with missing required data
df.dropna(subset=['day_of_week_enc', 'slot_type_enc', 'hour_of_day', 'is_event_day', 'wait_time_minute'], inplace=True)

# Define features
features = ['day_of_week_enc', 'slot_type_enc', 'hour_of_day', 'is_event_day']

# ======================== REGRESSION (Wait Time) ========================
print("\n===== Wait Time Models Evaluation =====")

X_time = df[features]
y_time = df['wait_time_minute']

regressors = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

best_reg_model = None
best_r2_score = float('-inf')
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in regressors.items():
    try:
        neg_mse = cross_val_score(model, X_time, y_time, cv=kf, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-neg_mse).mean()
        mae = -cross_val_score(model, X_time, y_time, cv=kf, scoring='neg_mean_absolute_error').mean()
        r2 = cross_val_score(model, X_time, y_time, cv=kf, scoring='r2').mean()

        print(f"\n{name} Regression:")
        print(f"  Avg RMSE: {rmse:.2f}")
        print(f"  Avg MAE: {mae:.2f}")
        print(f"  Avg R² Score: {r2:.3f}")

        if r2 > best_r2_score:
            best_r2_score = r2
            best_reg_model = model
    except Exception as e:
        print(f"\n{name} failed with error: {e}")

# Retrain best regression model on full data
best_reg_model.fit(X_time, y_time)
joblib.dump(best_reg_model, 'wait_time_model.pkl')


# ======================== CLASSIFICATION (Slot Availability) ========================
print("\n===== Slot Availability Models Evaluation =====")

df['slot_available'] = (df['wait_time_minute'] == 0).astype(int)
X_clf = df[features]
y_clf = df['slot_available']

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

best_clf_model = None
best_acc = 0

for name, clf in classifiers.items():
    try:
        acc_scores = cross_val_score(clf, X_clf, y_clf, cv=kf, scoring='accuracy')
        avg_acc = acc_scores.mean()

        print(f"\n{name} Classifier:")
        print(f"  Accuracy (5-Fold Avg): {avg_acc:.3f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_clf_model = clf
    except Exception as e:
        print(f"\n{name} failed with error: {e}")

# Retrain best classifier model on full data
best_clf_model.fit(X_clf, y_clf)
joblib.dump(best_clf_model, 'availability_model.pkl')

# Save encoders
joblib.dump(le_day, 'labelencoder_day.pkl')
joblib.dump(le_slot, 'labelencoder_slot.pkl')

print("\nBest models saved successfully.")

