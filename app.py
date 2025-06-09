from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained models and encoders
wait_time_model = joblib.load('wait_time_model.pkl')
availability_model = joblib.load('availability_model.pkl')
le_day = joblib.load('labelencoder_day.pkl')
le_slot = joblib.load('labelencoder_slot.pkl')

@app.route('/predict', methods=['GET', 'POST'])

def predict_wait_time():
    
    if request.method == 'GET':
        return "✅ This route only works with POST. Use a tool like curl or Postman."
    data = request.get_json()

    try:
        # Extract input features from request JSON
        day_of_week = data['day_of_week']
        slot_type = data['slot_type']
        checkin_timestamp = data['checkin_timestamp']
        is_event_day = data['is_event_day']
        is_weekend = data['is_weekend']

        # Preprocess features
        hour_of_day = pd.to_datetime(checkin_timestamp).hour
        day_enc = le_day.transform([day_of_week])[0]
        slot_enc = le_slot.transform([slot_type])[0]
        event_flag = 1 if is_event_day else 0
        weekend_flag = 1 if is_weekend else 0

        input_df = pd.DataFrame({
            'day_of_week_enc': [day_enc],
            'slot_type_enc': [slot_enc],
            'hour_of_day': [hour_of_day],
            'is_event_day': [event_flag],
            'is_weekend': [weekend_flag]
        })

        # Predict wait time
        wait_time_pred = wait_time_model.predict(input_df)[0]

        return jsonify({
            'predicted_wait_time_minutes': round(wait_time_pred, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict_availability', methods=['POST'])
def predict_availability():
    data = request.get_json()

    try:
        # Extract input features from request JSON
        day_of_week = data['day_of_week']
        slot_type = data['slot_type']
        checkin_timestamp = data['checkin_timestamp']
        is_event_day = data['is_event_day']
        is_weekend = data['is_weekend']

        # Preprocess features
        hour_of_day = pd.to_datetime(checkin_timestamp).hour
        day_enc = le_day.transform([day_of_week])[0]
        slot_enc = le_slot.transform([slot_type])[0]
        event_flag = 1 if is_event_day else 0
        weekend_flag = 1 if is_weekend else 0

        input_df = pd.DataFrame({
            'day_of_week_enc': [day_enc],
            'slot_type_enc': [slot_enc],
            'hour_of_day': [hour_of_day],
            'is_event_day': [event_flag],
            'is_weekend': [weekend_flag]
        })

        # Predict slot availability probability
        probability = availability_model.predict_proba(input_df)[0][1]

        return jsonify({
            'probability_slot_available': round(probability, 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

