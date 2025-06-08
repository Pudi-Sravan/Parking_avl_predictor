import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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

# Drop rows with any NaNs
df.dropna(subset=['day_of_week_enc', 'slot_type_enc', 'hour_of_day', 'is_event_day', 'wait_time_minute'], inplace=True)

# ========== Wait Time (Regression) ==========
print("\n===== Wait Time Models Evaluation =====")
features = ['day_of_week_enc', 'slot_type_enc', 'hour_of_day', 'is_event_day']
X_time = df[features]
y_time = df['wait_time_minute']
X_train, X_test, y_train, y_test = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

regressors = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

best_reg_model = None
best_r2 = float('-inf')

for name, model in regressors.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{name} Regression:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R² Score: {r2:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_reg_model = model
    except Exception as e:
        print(f"\n{name} failed with error: {e}")

joblib.dump(best_reg_model, 'wait_time_model.pkl')

# ========== Slot Availability (Classification) ==========
print("\n===== Slot Availability Models Evaluation =====")
df['slot_available'] = (df['wait_time_minute'] == 0).astype(int)

X_clf = df[features]
y_clf = df['slot_available']
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

best_clf_model = None
best_clf_acc = 0

for name, clf in classifiers.items():
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{name} Classifier:")
        print(f"  Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        if acc > best_clf_acc:
            best_clf_acc = acc
            best_clf_model = clf
    except Exception as e:
        print(f"\n{name} failed with error: {e}")

joblib.dump(best_clf_model, 'availability_model.pkl')
joblib.dump(le_day, 'labelencoder_day.pkl')
joblib.dump(le_slot, 'labelencoder_slot.pkl')

print("\n✅ Best models saved successfully.")

