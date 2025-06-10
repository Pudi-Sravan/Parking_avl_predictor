import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# ======================== Load and Preprocess Data ========================

df = pd.read_csv('info.csv')

df['checkin_timestamp'] = pd.to_datetime(df['checkin_timestamp'].astype(str).str.strip(), errors='coerce')
df['checkout_timestamp'] = pd.to_datetime(df['checkout_timestamp'].astype(str).str.strip(), errors='coerce')

print("Checkin parsing errors:", df['checkin_timestamp'].isna().sum())
print("Checkout parsing errors:", df['checkout_timestamp'].isna().sum())

df['hour_of_day'] = df['checkin_timestamp'].dt.hour
df['duration_minutes'] = (df['checkout_timestamp'] - df['checkin_timestamp']).dt.total_seconds() / 60

# Add derived features
df['is_weekend'] = df['checkin_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)  # 5 = Saturday, 6 = Sunday
df['is_event_day'] = df['is_event_day'].astype(int)

# ======================== Label Encoding ========================

le_day = LabelEncoder()
le_slot = LabelEncoder()
le_event = LabelEncoder()

df['day_of_week_enc'] = le_day.fit_transform(df['day_of_week'])
df['slot_type_enc'] = le_slot.fit_transform(df['slot_type'])
df['event_type_enc'] = le_event.fit_transform(df['event_type'])

df.dropna(subset=['day_of_week_enc', 'slot_type_enc', 'event_type_enc', 'hour_of_day', 'duration_minutes'], inplace=True)

# ======================== Feature Set ========================

features = [
    'day_of_week_enc',
    'slot_type_enc',
    'event_type_enc',
    'hour_of_day',
    'is_event_day',
    'is_weekend'
]

# ======================== Wait Time Prediction ========================

print("\n===== Wait Time Prediction (Regression) =====")
X_time = df[features]
y_time = df['duration_minutes']

regressors = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Huber Regressor": HuberRegressor()
}

best_r2 = float('-inf')
best_reg_model = None
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in regressors.items():
    try:
        r2 = cross_val_score(model, X_time, y_time, cv=kf, scoring='r2').mean()
        rmse = np.sqrt(-cross_val_score(model, X_time, y_time, cv=kf, scoring='neg_mean_squared_error')).mean()
        mae = -cross_val_score(model, X_time, y_time, cv=kf, scoring='neg_mean_absolute_error').mean()

        print(f"{name}: R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_reg_model = model
    except Exception as e:
        print(f"{name} failed: {e}")

if best_reg_model is not None:
    best_reg_model.fit(X_time, y_time)
    joblib.dump(best_reg_model, 'wait_time_model.pkl')

# ======================== Slot Availability Prediction ========================

print("\n===== Slot Availability Prediction (Classification) =====")
df["is_slot_available"] = df["wait_time_minute"].apply(lambda x: 1 if x <= 5 else 0)

X_clf = df[["day_of_week", "slot_type", "event_type", "is_event_day"]].copy()
y_clf = df["is_slot_available"]

# Encode using same encoders
X_clf["day_of_week"] = le_day.transform(X_clf["day_of_week"])
X_clf["slot_type"] = le_slot.transform(X_clf["slot_type"])
X_clf["event_type"] = le_event.transform(X_clf["event_type"])

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, zero_division=0),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'roc_auc': 'roc_auc'
}

best_f1 = 0
best_clf = None
kf_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    try:
        scores = cross_validate(clf, X_clf, y_clf, cv=kf_clf, scoring=scoring)

        print(f"\n{name}:")
        print(f"  Accuracy: {np.mean(scores['test_accuracy']):.3f}")
        print(f"  F1 Score: {np.mean(scores['test_f1']):.3f}")
        print(f"  Precision: {np.mean(scores['test_precision']):.3f}")
        print(f"  Recall: {np.mean(scores['test_recall']):.3f}")
        print(f"  ROC AUC: {np.mean(scores['test_roc_auc']):.3f}")

        if np.mean(scores['test_f1']) > best_f1:
            best_f1 = np.mean(scores['test_f1'])
            best_clf = clf
    except Exception as e:
        print(f"{name} failed: {e}")

if best_clf is not None:
    best_clf.fit(X_clf, y_clf)
    joblib.dump(best_clf, 'availability_model.pkl')

# ======================== Save Encoders ========================

joblib.dump(le_day, 'labelencoder_day.pkl')
joblib.dump(le_slot, 'labelencoder_slot.pkl')
joblib.dump(le_event, 'labelencoder_event.pkl')

print("\n✅ Models and encoders saved successfully!")
print("Best Classifier:", best_clf)
print("Best Regressor:", best_reg_model)
