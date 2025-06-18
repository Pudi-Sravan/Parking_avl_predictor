from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# =================== Load Models and Encoders ===================

wait_time_model = joblib.load('wait_time_model.pkl')
availability_model = joblib.load('availability_model.pkl')
le_day = joblib.load('labelencoder_day.pkl')
le_slot = joblib.load('labelencoder_slot.pkl')
le_event = joblib.load('labelencoder_event.pkl')

# =================== Utility Function ===================

def encode_input(data):
    try:
        # Parse timestamp to get hour and weekend info
        checkin_ts = pd.to_datetime(data['checkin_timestamp'])
        hour_of_day = checkin_ts.hour
        is_weekend = 1 if checkin_ts.dayofweek in [5, 6] else 0

        # Encode categorical features
        day_enc = le_day.transform([data['day_of_week']])[0]
        slot_enc = le_slot.transform([data['slot_type']])[0]
        event_enc = le_event.transform([data['event_type']])[0]

        return {
            "day_of_week_enc": day_enc,
            "slot_type_enc": slot_enc,
            "event_type_enc": event_enc,
            "hour_of_day": hour_of_day,
            "is_event_day": int(data['is_event_day']),
            "is_weekend": is_weekend
        }, None

    except Exception as e:
        return None, str(e)

# =================== Routes ===================

@app.route('/predict', methods=['POST'])
def predict_wait_time():
    data = request.get_json()
    encoded, error = encode_input(data)

    if error:
        return jsonify({'error': error}), 400

    input_df = pd.DataFrame([encoded])
    try:
        wait_time_pred = wait_time_model.predict(input_df)[0]
        return jsonify({'predicted_wait_time_minutes': round(wait_time_pred, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_availability', methods=['POST'])
def predict_availability():
    data = request.get_json()
    encoded, error = encode_input(data)

    if error:
        return jsonify({'error': error}), 400

    input_df = pd.DataFrame([encoded])
    try:
        probability = availability_model.predict_proba(input_df)[0][1]
        return jsonify({'probability_slot_available': round(probability, 3)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/newdata', methods=['POST'])
def log_new_data():
    data = request.get_json()

    required_fields = [
        "slot_id",
        "checkin_timestamp",
        "checkout_timestamp",
        "day_of_week",
        "slot_type",
        "event_type",
        "is_event_day",
        "wait_time_minute"
    ]

    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Prepare row for CSV
    row = {
        "slot_id": data["slot_id"],
        "checkin_timestamp": data["checkin_timestamp"],
        "checkout_timestamp": data["checkout_timestamp"],
        "day_of_week": data["day_of_week"],
        "slot_type": data["slot_type"],
        "event_type": data["event_type"],
        "is_event_day": int(data["is_event_day"]),
        "wait_time_minute": float(data["wait_time_minute"])
    }

    # Save to CSV
    try:
        df = pd.DataFrame([row])
        file_exists = os.path.isfile("walmart_parking_data_jan.csv")
        df.to_csv("walmart_parking_data_jan.csv", mode='a', header=not file_exists, index=False)
        return jsonify({"message": "Data appended to CSV successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save data: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    return "Flask is working!"


# =================== Run ===================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


