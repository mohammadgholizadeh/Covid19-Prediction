import numpy as np
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import math
import csv
from datetime import datetime
import datetime

target = ['total_cases',
          'total_deaths',
          'total_vaccinations',
          'people_vaccinated',
          'people_fully_vaccinated',
          'new_cases',
          'new_deaths'
          ]
target_day = [7, 14, 21, 28, 30]
locations = ['Austria', 'Argentina', 'Canada', 'United States', 'Malaysia']
df = pd.read_csv("pre_processed_dataset_V1.3.csv")

for con in locations:
    for tar in target:
        for day in target_day:
            df = df[df["location"] == con]
            df['year'] = df['date'].str[-4:]
            df['month'] = df['date'].str[:2]
            df['day'] = df['date'].str[3:5]

            # start = datetime.datetime.strptime("01-12-2021", "%d-%m-%Y")
            # end = datetime.datetime.strptime("31-12-2021", "%d-%m-%Y")
            start = datetime.date(2020, 1, 1)
            end = datetime.date(2020, 2, 1)
            date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
            # dt_obj = datetime.fromtimestamp(date_generated).strftime('%d-%m-%y')
            # time_series_array = pd.date_range('2020-01-12', periods=30)
            time_series_array = date_generated
            df_forecast = pd.DataFrame(columns=["date"])
            df_forecast.loc[:, "date"] = time_series_array
            ll = []
            for i in time_series_array:
                ll.append(str(i))

            df_forecast.loc[:, "date"] = ll
            df_forecast['year'] = df_forecast['date'].str[:4]
            df_forecast['month'] = df_forecast['date'].str[5:7]
            df_forecast['day'] = df_forecast['date'].str[8:10]

            x = df[['year', 'month', 'day']]
            y = df[tar]

            for_train = len(x) - 7
            x_train = x.iloc[:for_train]
            y_train = y.iloc[:for_train]
            x_test = x.iloc[for_train:]
            y_test = y.iloc[for_train:]

            clf = RandomForestRegressor(n_estimators=220)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_test = y_test.to_numpy()

            rae = np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test - np.mean(y_test))))
            rrse = np.sqrt(np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test))))
            print("R2:", r2_score(y_test, y_pred))
            print("Mean absolute error:", mean_absolute_error(y_test, y_pred))
            print("Mean squared error:", mean_squared_error(y_test, y_pred))
            print("Root mean square error:", math.sqrt(mean_squared_error(y_test, y_pred)))
            print("Relative absolute error:", rae)
            print("Root relative squared error:", rrse)

            pyplot.plot(y_test, label='Expected')
            pyplot.plot(y_pred, label='Predicted')
            pyplot.legend()
            pyplot.show()



#ax = x_train.plot()
#ax1 = y_pred.plot(ax=ax)
#y_test.plot(ax=ax1)
