import pandas as pd
import matplotlib.pyplot as plt
file_path = 'C:\Users\91779\Desktop\OIBSIP\OIBSIP\task2\archive2.csv'  
df = pd.read_csv(file_path)
print(df.head())
print(df.columns)
plt.figure(figsize=(10, 6))
plt.plot(df[' Date'], df[' Estimated Unemployment Rate (%)'], marker='o', linestyle='-')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()
print(df.describe())
df[' Date'] = pd.to_datetime(df[' Date'])
df.set_index(' Date', inplace=True)
monthly_mean = df.resample('M').mean()  
print(monthly_mean)
correlation_matrix = df.corr()
print(correlation_matrix)
