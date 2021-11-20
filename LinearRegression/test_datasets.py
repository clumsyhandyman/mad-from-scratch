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

    print(df.head(10))

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


def test_learning_rate(dataset, plot=True):
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

    lr_list = [0.2, 0.1, 0.001]
    mse_log_list = []
    for i in range(len(lr_list)):
        learner = LinearRegression(x_train, y_train, plot=plot, verbose=True)
        w, mse_log = learner.gradient_descend(learner.x_train, learner.y_train, lr=lr_list[i])
        mse_log_list.append(mse_log)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=100)
    ax.set_title('Learning curves of different learning rates')
    ax.plot(mse_log_list[0][:10], marker='.', label=f'lr = {lr_list[0]}')
    ax.plot(mse_log_list[1][:200], label=f'lr = {lr_list[1]}')
    ax.plot(mse_log_list[2][:200], linestyle='dashed', label=f'lr = {lr_list[2]}')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()



# test('uscrime')
# test('BostonHousing')
test('diamonds')

# test_learning_rate('BostonHousing')

