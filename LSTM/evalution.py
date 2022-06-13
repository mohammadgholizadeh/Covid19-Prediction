import pandas as pd
import matplotlib.pyplot as plt

target = 'new_cases'
df = pd.read_csv('evalution.csv')
tg = df[['R2']]
tg = tg.set_index(df['target'])
tg.plot.bar()
plt.savefig('R2.jpg', dpi = 1000, transparent=True)
