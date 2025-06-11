import pandas as pd

# Load your CSV file
df = pd.read_csv('walmart_parking_data_jan (1).csv')

# Rename columns for clarity
df.columns = ['slot_id', 'checkin_timestamp', 'checkout_timestamp', 'day_of_week', 
              'slot_type', 'event_type', 'is_event_day', 'wait_time_minute']

# Convert checkin_timestamp to datetime format
df['checkin_timestamp'] = pd.to_datetime(df['checkin_timestamp'])

# Sort the DataFrame by checkin_timestamp
df_sorted = df.sort_values(by='checkin_timestamp')

# Save the sorted DataFrame to a new CSV
df_sorted.to_csv('sorted_walmart_data.csv', index=False)
