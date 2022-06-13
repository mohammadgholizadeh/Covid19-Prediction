import pandas as pd
import matplotlib.pyplot as plt


targets = ['R2', 'MAE', 'MSE', 'RMSE', 'RAE', 'RRSE']
for target in targets:
    df = pd.read_csv('evalution.csv')
    tg = df[[target]]
    tg = tg.set_index(df['target'])
    tg.plot.bar()
    plt.savefig(target + '.jpg', dpi = 1000, transparent=True)
