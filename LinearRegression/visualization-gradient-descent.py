import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression

np.random.seed(seed=42)

n = 51
x = np.linspace(0, 100, n)

y = np.zeros((3, n))

y[0] = np.random.randn(n) * 10 + 10

y[1] = x * (.2 + 0.1 * np.random.randn(n)) + 2 * np.random.randn(n)
y[1, :n//4] = y[1, :n//4] + 8 * np.random.randn(n//4)
y[1, n*3//4:] = y[1, n*3//4:] + 8 * np.random.randn(len(y[1, n*3//4:]))

y[2] = x * .3 + 2 * np.random.randn(n)

x = np.transpose(np.atleast_2d(x))
y = np.transpose(y[2])
print(x.shape)
print(y.shape)

# learner = LinearRegression(x, y, plot=True)
# learner.fit(lr=0.2)
# y_hat = learner.predict(x)


raw_x = x.copy()
x_train = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
y_train = y

x = x_train
y = y_train

n = x.shape[0]
d = x.shape[1]
w = np.zeros(d)
mse_log = []

lr = [0.007, 0.0001]

ss_total = np.sum((y - np.mean(y)) ** 2)

for i in range(5):
    print(f'Iteration: {i}')
    y_pred = np.matmul(x, w)

    sse = np.sum((y - y_pred) ** 2)
    mse = np.mean((y - y_pred) ** 2)
    ss_error = np.sum((y - y_pred) ** 2)
    r_square = 1 - ss_error / ss_total

    mse_log.append(mse)

    gradient_w = -2 * np.matmul((y - y_pred), x) / n

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    ax.set_title(f'Iteration: {i} \n' + r'$SS_E =$' + f' {sse:.2f}, ' + r'$R^2 =$' + f' {r_square*100:.1f}%\n'
                 + r'$\frac{\partial MSE}{\partial w} = $' + f'{gradient_w[0]:.3f}, {gradient_w[1]:.3f}')
    ax.set_xlabel('x')
    ax.plot(raw_x, y, marker='o', markersize=3, linestyle='None', alpha=0.7,
             label='y', color='#2ca02c')
    ax.plot(raw_x, y_pred, marker='o', markersize=3, lw='.5', alpha=0.7,
            label=r'$\hat{y} =$' + f'{w[0]:.3f} + {w[1]:.3f} x', color='#d62728')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'document/figures/visualize-{i}.png')


    w -= lr * gradient_w
    print(f'w = {w}')
    print()
#
# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=100)
# ax.plot(mse_log, marker='o', markersize=3, alpha=0.7, color='#2ca02c')
# plt.tight_layout()
# plt.show()







