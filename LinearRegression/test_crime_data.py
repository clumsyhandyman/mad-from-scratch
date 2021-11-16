import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression

df = pd.read_csv('uscrime.txt', sep='\t')
print(df.shape)
print(df.head())
df = df.to_numpy()
print(df.shape)
x_train = df[0:30, 0: -1]
y_train = df[0:30, -1]

x_test = df[30:, 0:-1]
y_test = df[30:, -1]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

leaner = LinearRegression(x_train, y_train)
leaner.fit(plot=True)
y_pred_train = leaner.predict(x_train)
y_pred_test = leaner.predict(x_test)


fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=100, sharex='all', sharey='all')
axes[0].set_title('Training')
axes[0].plot(y_train, y_pred_train, marker='o', markersize=4, linestyle='None', color='#1f77b4')
axes[0].axline((np.mean(y_train), np.mean(y_train)), slope=1., color='red')
axes[0].set_ylabel('Predicted value')
axes[0].set_xlabel('True value')

axes[1].set_title('Testing')
axes[1].plot(y_test, y_pred_test, marker='o', markersize=4, linestyle='None', color='#2ca02c')
axes[1].axline((np.mean(y_test), np.mean(y_test)), slope=1., color='red')
axes[1].set_ylabel('Predicted value')
axes[1].set_xlabel('True value')

plt.tight_layout()
plt.show()
