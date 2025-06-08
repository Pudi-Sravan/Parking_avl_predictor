import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# Load your dataset
df = pd.read_csv('info.csv')

# Clean and parse timestamps
df['checkin_timestamp'] = df['checkin_timestamp'].astype(str).str.strip()
df['checkout_timestamp'] = df['checkout_timestamp'].astype(str).str.strip()

df['checkin_timestamp'] = pd.to_datetime(df['checkin_timestamp'], errors='coerce')
df['checkout_timestamp'] = pd.to_datetime(df['checkout_timestamp'], errors='coerce')

print("Checkin parsing errors:", df['checkin_timestamp'].isna().sum())
print("Checkout parsing errors:", df['checkout_timestamp'].isna().sum())

# Extract hour of day from checkin
df['hour_of_day'] = df['checkin_timestamp'].dt.hour

# Encode categorical features
le_day = LabelEncoder()
df['day_of_week_enc'] = le_day.fit_transform(df['day_of_week'])

le_slot = LabelEncoder()
df['slot_type_enc'] = le_slot.fit_transform(df['slot_type'])

# Convert is_event_day to binary
df['is_event_day'] = df['is_event_day'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

# --- Wait Time Model ---
features = ['day_of_week_enc', 'slot_type_enc', 'hour_of_day', 'is_event_day']
X_time = df[features]
y_time = df['wait_time_minute']

X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

wait_time_model = RandomForestRegressor(n_estimators=100, random_state=42)
wait_time_model.fit(X_train_time, y_train_time)

y_pred_time = wait_time_model.predict(X_test_time)
mse = mean_squared_error(y_test_time, y_pred_time)
print(f"Wait Time Model - Test Mean Squared Error: {mse:.2f}")

# --- Slot Availability Model ---
df['slot_available'] = (df['wait_time_minute'] == 0).astype(int)

X_avail = df[features]
y_avail = df['slot_available']

X_train_avail, X_test_avail, y_train_avail, y_test_avail = train_test_split(X_avail, y_avail, test_size=0.2, random_state=42)

availability_model = RandomForestClassifier(n_estimators=100, random_state=42)
availability_model.fit(X_train_avail, y_train_avail)

# --- Save Everything ---
joblib.dump(wait_time_model, 'wait_time_model.pkl')
joblib.dump(availability_model, 'availability_model.pkl')
joblib.dump(le_day, 'labelencoder_day.pkl')
joblib.dump(le_slot, 'labelencoder_slot.pkl')

print("Models and encoders saved successfully.")
