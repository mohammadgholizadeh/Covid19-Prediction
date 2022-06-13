import csv  

header = ['', 'R2', 'MAE', 'MSE', 'RMSE', 'RAE', 'RRSE']

with open('evalution.csv', 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)