import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
import datetime


country = "Austria"
target = 'new_cases'
df_confirmed = pd.read_csv("pre_processed_dataset.csv")
df_confirmed_country = df_confirmed[df_confirmed["location"] == country]
df_confirmed_country = pd.DataFrame(df_confirmed_country[['date', target]])
df_confirmed_country = df_confirmed_country.set_index('date')
df_confirmed_country = df_confirmed_country.rename_axis(None)

df_confirmed_country.index = pd.to_datetime(df_confirmed_country.index, infer_datetime_format=True)

# df_confirmed_country.plot(figsize=(10, 5), title="COVID confirmed cases")

print("Total days in the dataset", len(df_confirmed_country))

# Use data until 14 days before as training
x = len(df_confirmed_country) - 0

train = df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

# scale or normalize data as the data is too skewed
scaler = MinMaxScaler()
scaler.fit(train)

train_scaled = scaler.transform(train)
#test_scaled = scaler.transform(test)

# Use TimeSeriestrain_generator to generate data in sequences.
seq_size = 11  # number of steps (lookback)
n_features = 1  # number of features. This dataset is univariate so it is 1
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(train))
print("Total number of samples in the generated data = ", len(train_generator))

# Check data shape from generator
# x, y = train_generator[10]  # Check train_generator
# Takes 7 days as x and 8th day as y (for seq_size=7)

# Also generate test data
#test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(test))  # 14 as we're using last 14 days for test
#print("Total number of samples in the generated data = ", len(test_generator))  # 7
# Check data shape from generator
# x, y = test_generator[0]

# Define Model
model = Sequential()
model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(seq_size, n_features)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
print('Train...')
##########################

history = model.fit(train_generator,
                              #validation_data=test_generator,
                              epochs=50, steps_per_epoch=10)

# forecast
prediction = []  # Empty list to populate later with predictions

current_batch = train_scaled[-seq_size:]  # Final data points in train
current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

## Predict future, beyond test dates
future = 30
for i in range(future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

start = datetime.datetime.strptime("01-12-2021", "%d-%m-%Y")
end = datetime.datetime.strptime("31-12-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]


time_series_array = date_generated  # Get dates for test data

# Add new dates for the forecast period
#for k in range(0, future):
#    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Create a dataframe to capture the forecast data

df_forecast = pd.DataFrame(columns=[ "predicted"], index=time_series_array)
actual = pd.DataFrame(columns=["actual"], index=time_series_array)
pred = pd.DataFrame(columns=["predicted"], index=time_series_array)

df_forecast.loc[:, "predicted"] = rescaled_prediction[:, 0]
#df_forecast.loc[:, "actual_confirmed"] = test[target]
actual.loc[:, "actual"] = test[target]
pred.loc[:, "predicted"] = rescaled_prediction[:, 0]

df_forecast.plot(title="Predictions for next 30 days")
plt.savefig("new deaths vaccinated 30.jpg", dpi = 500)

ax = train.plot()
ax1 = pred.plot(ax=ax)
#actual.plot(ax=ax1)
#plt.savefig("new deaths vaccinated 30.jpg", dpi = 500)


