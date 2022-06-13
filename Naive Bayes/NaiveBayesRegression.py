import numpy as np
from matplotlib import pyplot
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import math
import csv
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
# 'Austria', 'Canada', 'United States', 'Malaysia', 'Argentina'
# 'Asia', 'Europe'
locations = ['Europe']
#df = pd.read_csv("pre_processed_dataset_V1.3.csv")
#df = pd.read_csv("Asia_pre_processed_dataset.csv")
df = pd.read_csv("Europe_pre_processed_dataset.csv")
future = 30

for con in locations:
    #df = df[df["location"] == con]
    df = df[df["continent"] == con]
    df['year'] = df['date'].str[-4:]
    df['month'] = df['date'].str[:2]
    df['day'] = df['date'].str[3:5]
    start = datetime.date(2021, 12, 1)
    end = datetime.date(2021, 12, future)
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
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

    for tar in target:
        x = df[['year', 'month', 'day']]
        y = df[tar]

        for day in target_day:
            for_train = len(x) - day
            x_train = x.iloc[:for_train]
            y_train = y.iloc[:for_train]
            x_test = x.iloc[for_train:]
            y_test = y.iloc[for_train:]

            clf = GaussianNB()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_test = y_test.to_numpy()

            rae = np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test - np.mean(y_test))))
            rrse = np.sqrt(np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test))))
            # print("R2:", r2_score(y_test, y_pred))
            # print("Mean absolute error:", mean_absolute_error(y_test, y_pred))
            # print("Mean squared error:", mean_squared_error(y_test, y_pred))
            # print("Root mean square error:", math.sqrt(mean_squared_error(y_test, y_pred)))
            # print("Relative absolute error:", rae)
            # print("Root relative squared error:", rrse)

            result = []
            result.append(tar + " in " + str(day))
            result.append(r2_score(y_test, y_pred))
            result.append(mean_absolute_error(y_test, y_pred))
            result.append(mean_squared_error(y_test, y_pred))
            #result.append(math.sqrt(mean_squared_error(y_test, y_pred)))
            result.append(0)
            result.append(rae)
            result.append(rrse)
            #'''
            with open('evalution.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(result)
            result.clear()
            
            pyplot.plot(y_test, label='Expected')
            pyplot.plot(y_pred, label='Predicted')
            pyplot.legend()
            pyplot.title("Prediction for " + str(day) + " days")
            pyplot.savefig(con +" "+ tar + " Predictions for next " + str(day) + " days.jpg", dpi=500)
            pyplot.show()
            

'''
for tar in target:
    x = df[['year', 'month', 'day']]
    y = df[tar]
    for_train = len(x) - 0
    x_train = x.iloc[:for_train]
    y_train = y.iloc[:for_train]
    x_test = df_forecast[['year', 'month', 'day']]
    y_test = y.iloc[for_train:]
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_test = y_test.to_numpy()
    #pyplot.plot(y_test, label='Expected')
    pyplot.plot(y_pred, label='Predicted')
    pyplot.legend()
    pyplot.title("Prediction for " + str(future) + " days(future)")
    pyplot.savefig(con + tar + " Predictions for next " + str(future) + " days(future).jpg", dpi=500)
    pyplot.show()
#'''







#ax = x_train.plot()
#ax1 = y_pred.plot(ax=ax)
#y_test.plot(ax=ax1)
