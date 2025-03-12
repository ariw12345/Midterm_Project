#############################################################################################
# Data Import 

import pandas as pd 
import os
import matplotlib.pyplot as plt
import glob


folder_path = 'C:/Users/ariwe/Downloads/midterm_project_data' # specify folder where data is located
all_files = os.listdir(folder_path)
metered_files = [file for file in all_files if 'metered' in file and file.endswith('.csv')]

# Initialize an empty list to hold the dataframes
dfs = []

# Load each file and append it to the dfs list
for file in metered_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes in the list into one large dataframe
combined_df = pd.concat(dfs, ignore_index=True)

print(combined_df)

metered_data = combined_df[['datetime_beginning_ept', 'mw']]

metered_data['datetime_beginning_ept'] = pd.to_datetime(metered_data['datetime_beginning_ept'])

#############################################################################################
# LSTM Part

import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Our target predictor is mw 

# Filter training set (2020-2023)
training_set = metered_data[metered_data['year'] < 2024]['mw'].values.reshape(-1, 1)

# Filter test set (2024)
test_set = metered_data[metered_data['year'] == 2024]['mw'].values.reshape(-1, 1)


# Feature Scaling (Normalization)
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)

# Create Time-Series Sequences for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])  # Past 60 time steps
        y.append(data[i, 0])  # Next value to predict
    return np.array(X), np.array(y)

# Prepare Training Data
X_train, y_train = create_sequences(training_set_scaled, time_steps=60)

# Reshape for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 10, batch_size = 32)
# Define the path where you want to save the model
save_path = r'C:/Users/ariwe/Downloads/midterm_project_data'

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Save the model to the specified directory
model.save(os.path.join(save_path, 'lstm_model.h5'))


# Train-Test Split Using Year-Based Filtering
dataset_train = metered_data[metered_data['year'] < 2024][['mw']]  # Training data (2020-2023)
dataset_test = metered_data[metered_data['year'] == 2024][['mw']]  # Test data (2024)

dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)  # Normalize data

# Create test set with 60 previous time steps for each prediction
X_test = []
for i in range(60, len(inputs)):  # Loop from 60 to the length of inputs
    X_test.append(inputs[i - 60:i, 0])  # Add 60 time steps

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape for LSTM

print(X_test.shape)

print(len(X_test))

# Make Predictions
predicted_mw_scaled = model.predict(X_test)
predicted_mw = sc.inverse_transform(predicted_mw_scaled)  # Convert back to original MW scale


metered_data['datetime_beginning_ept'] = pd.to_datetime(metered_data['datetime_beginning_ept'])

metered_data['year'] = metered_data['datetime_beginning_ept'].dt.year
metered_data['month'] = metered_data['datetime_beginning_ept'].dt.month
metered_data['day'] = metered_data['datetime_beginning_ept'].dt.day
metered_data['hour'] = metered_data['datetime_beginning_ept'].dt.hour
# Ensure test set alignment (offset by time steps)
actual_dates = metered_data[metered_data['year'] == 2024]['datetime_beginning_ept']  # Skip first 60


###################################################################################

# Load in PJM Forecast

file_path = 'Downloads\midterm_project_data\dom_pjm_hist_forecast_2024.csv'

# Import CSV file into a DataFrame
df = pd.read_csv(file_path)

pjm_forecast = df[['forecast_hour_beginning_ept', 'forecast_load_mw']]
pjm_forecast_first_instance = pjm_forecast.drop_duplicates(subset='forecast_hour_beginning_ept', keep='first')

pjm_forecast_first_instance['forecast_hour_beginning_ept'] = pd.to_datetime(pjm_forecast_first_instance['forecast_hour_beginning_ept'])

######################################################################################

# Visualizing the results
plt.figure(figsize=(10, 5))
plt.plot(actual_dates, dataset_test.values, color='red', label='Actual MW Demand')
plt.plot(actual_dates, predicted_mw, color='blue', label='Predicted MW Demand')
plt.plot(pjm_forecast_first_instance['forecast_hour_beginning_ept'], 
         pjm_forecast_first_instance['forecast_load_mw'], 
         color='green', label='PJM Forecast')
plt.xticks(rotation=45)
plt.title('Energy Demand Forecasting (2024)')
plt.xlabel('Time')
plt.ylabel('MW Demand')
plt.legend()
plt.show()

########################################################################################

# Plottting Individual Months

combined = pd.DataFrame({
    'actual_dates': actual_dates,
    'actual_mw': dataset_test.values.flatten(),  # Flatten if it's a 2D array
    'predicted_mw': predicted_mw.flatten(),     # Flatten if it's a 2D array
})

# Merge the PJM forecast data with the combined DataFrame based on matching 'forecast_hour_beginning_ept'
combined = pd.merge(combined, pjm_forecast_first_instance[['forecast_hour_beginning_ept', 'forecast_load_mw']], 
                    how='left', left_on='actual_dates', right_on='forecast_hour_beginning_ept')

# Drop the 'forecast_hour_beginning_ept' column as it's not needed for plotting
combined = combined.drop(columns=['forecast_hour_beginning_ept'])

combined['actual_dates'] = pd.to_datetime(combined['actual_dates'])

# Extract day, hour, and month into new columns
combined['day'] = combined['actual_dates'].dt.day
combined['hour'] = combined['actual_dates'].dt.hour
combined['month'] = combined['actual_dates'].dt.month


for month in range(1, 13):
    # Filter data for the specific month
    month_data = combined[combined['month'] == month]
    
    # Plot the graph for the month
    plt.figure(figsize=(10, 5))
    plt.plot(month_data['actual_dates'], month_data['actual_mw'], color='red', label='Actual MW Demand')
    plt.plot(month_data['actual_dates'], month_data['predicted_mw'], color='blue', label='Predicted MW Demand')
    plt.plot(month_data['actual_dates'], month_data['forecast_load_mw'], color='green', label='PJM Forecast')

    # Customize plot
    plt.xticks(rotation=45)
    plt.title(f'Energy Demand Forecasting - Month {month} (2024)')
    plt.xlabel('Time')
    plt.ylabel('MW Demand')
    plt.legend()

    # Show the plot
    plt.show()
print(combined)


# Model Evaluation 

combined_clean = combined.dropna(subset=['actual_mw', 'predicted_mw', 'forecast_load_mw'])

# Calculate MAE and RMSE for your model
mae_model = mean_absolute_error(combined_clean['actual_mw'], combined_clean['predicted_mw'])
rmse_model = np.sqrt(mean_squared_error(combined_clean['actual_mw'], combined_clean['predicted_mw']))

# Calculate MAE and RMSE for PJM forecast
mae_pjm = mean_absolute_error(combined_clean['actual_mw'], combined_clean['forecast_load_mw'])
rmse_pjm = np.sqrt(mean_squared_error(combined_clean['actual_mw'], combined_clean['forecast_load_mw']))

print(f"Model MAE: {mae_model:.2f}")
print(f"Model RMSE: {rmse_model:.2f}")
print(f"PJM Forecast MAE: {mae_pjm:.2f}")
print(f"PJM Forecast RMSE: {rmse_pjm:.2f}")

combined['model_residual'] = combined['actual_mw'] - combined['predicted_mw']
combined['pjm_residual'] = combined['actual_mw'] - combined['forecast_load_mw']

# Plot residuals
plt.figure(figsize=(10, 5))

# Plot the residuals for your model
plt.plot(combined['actual_dates'], combined['model_residual'], color='blue', label='Model Residual (Actual - Predicted)', alpha=0.6)

# Plot the residuals for the PJM forecast
plt.plot(combined['actual_dates'], combined['pjm_residual'], color='green', label='PJM Residual (Actual - PJM Forecast)', alpha=0.6)

plt.xticks(rotation=45)
plt.title('Residuals: Model vs PJM Forecast (Energy Demand)')
plt.xlabel('Time')
plt.ylabel('Residuals (MW)')
plt.legend()
plt.show()

print(combined)


combined.to_excel('combined_forecast_data.xlsx', index=False)
print(os.getcwd())


##########################
plt.figure(figsize=(10, 6))
plt.hist(combined['actual_mw'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Actual MW Demand')
plt.xlabel('MW Demand')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()