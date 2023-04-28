import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('boston_house.csv')

# print(df.head())
#
# print("CRIM min:", df['CRIM'].min())
print("CRIM mean:", df['CRIM'].mean())
print("CRIM median:", df['CRIM'].median())
# print("CRIM mode:", df['CRIM'].mode())
print("CRIM 0.25:", df['CRIM'].quantile(0.25))
print("CRIM 0.75:", df['CRIM'].quantile(0.75))
print("CRIM 0.99:", df['CRIM'].quantile(0.99))
# print("CRIM max:", df['CRIM'].max())
# print("DF mean:\n", df.mean())
# plt.show()
sb.set_theme()
crime = sb.histplot(df['CRIM'], kde=True)
plt.show()

# df['crime_log'] = np.log(df['CRIM'])
#
# sb.boxplot(data=df, x='crime_log')
#
# plt.show()


