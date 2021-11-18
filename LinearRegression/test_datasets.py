import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression


def test(dataset, plot=True):
    if dataset == 'uscrime':
        df = pd.read_csv('data/uscrime.txt', sep='\t')
    elif dataset == 'BostonHousing':
        df = pd.read_csv('data/BostonHousing.txt', sep=',')
    elif dataset == 'diamonds':
        df = pd.read_csv('data/diamonds.csv', sep=' ')
    else:
        print('No data set under the input name.')
        return

    df = df.to_numpy()
    x = df[:, 0:-1]
    y = df[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    learner = LinearRegression(x_train, y_train, plot=plot)
    learner.fit()

    y_pred_train = learner.predict(x_train)
    y_pred_test = learner.predict(x_test)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=100, sharex='all', sharey='all')
    axes[0].set_title(f'{dataset} Training')
    axes[0].plot(y_train, y_pred_train, marker='o', markersize=4, linestyle='None', color='#1f77b4')
    axes[0].axline((np.mean(y_train), np.mean(y_train)), slope=1., color='red')
    axes[0].set_ylabel('Predicted value')
    axes[0].set_xlabel('True value')

    axes[1].set_title(f'{dataset} Testing')
    axes[1].plot(y_test, y_pred_test, marker='o', markersize=4, linestyle='None', color='#2ca02c')
    axes[1].axline((np.mean(y_test), np.mean(y_test)), slope=1., color='red')
    axes[1].set_xlabel('True value')

    plt.tight_layout()
    plt.show()


# test('uscrime')
test('BostonHousing')
# test('diamonds')