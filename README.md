# Parking Availability Predictor

A machine learning-based system to predict parking slot availability and wait times for Walmart parking lots using historical data patterns.

## Features

- **Slot Availability Prediction**: Predicts whether parking slots will be available
- **Wait Time Estimation**: Estimates expected wait times for different slot types
- **Multi-slot Support**: Handles car slots (s1-s3), bike slots (s4), and disabled-accessible slots (s5)
- **Event-aware**: Considers special events and holidays that affect parking patterns
- **REST API**: Flask-based web service for real-time predictions

## Setup

### 1. Virtual Environment Setup

Create and activate a virtual environment to isolate project dependencies:

#### On Windows:
```bash
# Create virtual environment
python -m venv parking_env

# Activate virtual environment
parking_env\Scripts\activate
```

#### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv parking_env

# Activate virtual environment
source parking_env/bin/activate
```

### 2. Install Required Packages

Install all necessary Python packages using pip:

```bash
pip install pandas scikit-learn flask joblib numpy
```

#### Package Details:
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and utilities
- **flask**: Web framework for creating REST API
- **joblib**: Model serialization and persistence
- **numpy**: Numerical computing support

### 4. Train the Models

Run the model training script to create prediction models:

```bash
python model.py
```

This script will:
- Load the parking data from CSV
- Preprocess and encode categorical variables
- Train Random Forest models for availability and wait time prediction
- Save trained models and encoders to the `models/` directory
- Display model performance metrics

### 5. Start the Flask Server

Launch the web API server:

```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## API Usage

### Predict Parking Availability

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
    "slot_id": "s1",
    "day_of_week": "Monday",
    "hour": 14,
    "event_name": "Regular Day",
    "is_event_day": 0
}
```

**Response**:
```json
{
    "slot_id": "s1",
    "availability_prediction": 1,
    "wait_time_prediction": 3.5,
    "slot_type": "car"
}
```

## Model Information

- **Availability Model**: Random Forest Classifier predicting slot availability (0/1)
- **Wait Time Model**: Random Forest Regressor predicting wait time in minutes
- **Features Used**: slot_id, day_of_week, hour, event_name, is_event_day
- **Slot Types**: 
  - s1, s2, s3: Car parking slots
  - s4: Bike parking slot
  - s5: Disabled-accessible parking slot

## Data Format

The training data should include the following columns:
- `slot_id`: Parking slot identifier (s1-s5)
- `checkin_timestamp`: Entry timestamp
- `checkout_timestamp`: Exit timestamp  
- `day_of_week`: Day of the week
- `slot_type`: Type of parking slot (car/bike/disabled)
- `event_name`: Name of special event or "Regular Day"
- `is_event_day`: Binary flag for special events (0/1)
- `wait_time_minute`: Wait time in minutes

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Ensure virtual environment is activated and packages are installed
2. **Port already in use**: Change the port in `app.py` or stop other services using port 5000

### Deactivating Virtual Environment:
```bash
deactivate
```

## Future Enhancements

- Real-time data integration
- Advanced time series forecasting
- Mobile app integration
- Multi-location support
- Weather impact analysis
