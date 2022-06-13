import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
import csv


country = "Iran"
target = 'new_deaths'
#df_confirmed = pd.read_csv("Asia_pre_processed_dataset.csv")
df_confirmed = pd.read_csv("pre_processed_dataset_V1.3.csv")
df_confirmed_country = df_confirmed[df_confirmed["location"] == country]
df_confirmed_country = pd.DataFrame(df_confirmed_country[['date', target]])
df_confirmed_country = df_confirmed_country.set_index('date')
df_confirmed_country = df_confirmed_country.rename_axis(None)

df_confirmed_country.index = pd.to_datetime(df_confirmed_country.index, infer_datetime_format=True)
print("Total days in the dataset", len(df_confirmed_country))

target_day = 30
x = len(df_confirmed_country) - target_day

train = df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

# scale or normalize data as the data is too skewed
scaler = MinMaxScaler()
scaler.fit(train)

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

# Use TimeSeriestrain_generator to generate data in sequences.
seq_size = 6  # number of steps (lookback)
n_features = 1  # number of features. This dataset is univariate so it is 1
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(train))
print("Total number of samples in the generated data = ", len(train_generator))

# Check data shape from generator
# x, y = train_generator[10]  # Check train_generator
# Takes 7 days as x and 8th day as y (for seq_size=7)

# Also generate test data
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
print("Total number of samples in the original training data = ", len(test))  # 14 as we're using last 14 days for test
print("Total number of samples in the generated data = ", len(test_generator))  # 7
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
                              validation_data=test_generator,
                              epochs=50, steps_per_epoch=10)

# forecast
prediction = []  # Empty list to populate later with predictions

current_batch = train_scaled[-seq_size:]  # Final data points in train
current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

## Predict future, beyond test dates
future = 0
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = test.index  # Get dates for test data

# Add new dates for the forecast period
#for k in range(0, future):
#    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

# Create a dataframe to capture the forecast data

df_forecast = pd.DataFrame(columns=["actual_confirmed", "predicted"], index=time_series_array)
actual = pd.DataFrame(columns=["actual"], index=time_series_array)
pred = pd.DataFrame(columns=["predicted"], index=time_series_array)

df_forecast.loc[:, "predicted"] = rescaled_prediction[:, 0]
df_forecast.loc[:, "actual_confirmed"] = test[target]
actual.loc[:, "actual"] = test[target]
pred.loc[:, "predicted"] = rescaled_prediction[:, 0]

rae = np.sum(np.abs(actual.to_numpy() - pred.to_numpy())) / (np.sum(np.abs(actual.to_numpy() - np.mean(actual.to_numpy()))))
rrse = np.sqrt(np.sum(np.square(actual.to_numpy() - pred.to_numpy())) / np.sum(np.square(actual.to_numpy() - np.mean(actual.to_numpy()))))
print("R2:", r2_score(actual, pred))
print("Mean absolute error:", mean_absolute_error(actual, pred))
print("Mean squared error:", mean_squared_error(actual, pred))
print("Root mean square error:", math.sqrt(mean_squared_error(actual, pred)))
print("Relative absolute error:", rae)
print("Root relative squared error:", rrse)

result = []
result.append(target)
result.append(r2_score(actual, pred))
result.append(mean_absolute_error(actual, pred))
result.append(mean_squared_error(actual, pred))
result.append(math.sqrt(mean_squared_error(actual, pred)))
result.append(rae)
result.append(rrse)
#with open('prediction.csv', 'a', encoding='UTF8') as f:
#    writer = csv.writer(f)
#    writer.writerow(rescaled_prediction)

df_forecast.plot(title="Predictions for next " + str(target_day) +" days")
#plt.savefig(target + " Predictions for next " +str(target_day)+ " days.jpg", dpi =500)

ax = train.plot()
ax1 = pred.plot(ax=ax)
actual.plot(ax=ax1)
#plt.savefig(target + "total cases Predictions for next " +str(target_day)+ " days#.jpg", dpi =500)


