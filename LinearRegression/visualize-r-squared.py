import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression

np.random.seed(seed=42)

n = 51
x = np.linspace(0, 100, n)
print(x)

y = np.zeros((3, n))

y[0] = np.random.randn(n) * 10 + 10

y[1] = x * (.2 + 0.1 * np.random.randn(n)) + 2 * np.random.randn(n)
y[1, :n//4] = y[1, :n//4] + 8 * np.random.randn(n//4)
y[1, n*3//4:] = y[1, n*3//4:] + 8 * np.random.randn(len(y[1, n*3//4:]))

y[2] = x * .3 + 2 * np.random.randn(n)

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='all')
for i in range(3):
    axes[i].set_xlabel('x')
    axes[i].plot(x, y[i], marker='o', markersize=3, linestyle='None', alpha=0.7,
                 label=f'y{i+1}', color=color_list[i])
    axes[i].legend()
plt.tight_layout()
plt.savefig('document/figures/comparison-raw-data.png')

y_bar = np.zeros((3, n))
for i in range(3):
    y_bar[i] = np.ones(n) * np.mean(y[i])

w = []
y_hat = np.zeros((3, n))
for i in range(3):
    learner = LinearRegression(np.transpose(np.atleast_2d(x)), np.transpose(y[i]))
    learner.fit()
    w.append(learner.get_unscaled_weights())
    y_hat[i] = learner.predict(np.transpose(np.atleast_2d(x)))

ss_total = np.zeros(3)
ss_regression = np.zeros(3)
ss_error = np.zeros(3)

for i in range(3):
    ss_total[i] = np.sum((y[i] - np.mean(y[i])) ** 2)
    ss_regression[i] = np.sum((y_hat[i] - np.mean(y[i])) ** 2)
    ss_error[i] = np.sum((y[i] - y_hat[i]) ** 2)

r_square = 1 - ss_error / ss_total

fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='all')
for i in range(3):
    axes[i].set_title(r'$SS_T =$ ' + f'{ss_total[i]:.2f}')
    axes[i].set_xlabel('x')
    axes[i].plot(x, y[i], marker='o', markersize=3, linestyle='None', alpha=0.7,
                 label=f'y{i+1}', color=color_list[i])
    axes[i].plot(x, y_bar[i], marker='o', markersize=3, lw='.5', alpha=0.7,
                 label=r'$\bar{y}$' + f'{i + 1}', color='black')
    for j in range(n):
        axes[i].plot([x[j], x[j]], [y_bar[i, j], y[i, j]], lw='.5',
                     color=color_list[i])
    axes[i].legend()
plt.tight_layout()
plt.savefig('document/figures/comparison-sst.png')


fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='all')
for i in range(3):
    axes[i].set_title( r'$\hat{y} = $' + f'{w[i][0]:.2f} + {w[i][1]:.2f}x')
    axes[i].set_xlabel('x')
    axes[i].plot(x, y[i], marker='o', markersize=3, linestyle='None', alpha=0.7,
                 label=f'y{i+1}', color=color_list[i])
    axes[i].plot(x, y_hat[i], marker='o', markersize=3, lw='.5', alpha=0.7,
                 label=r'$\hat{y}$' + f'{i + 1}', color='#d62728')
    axes[i].legend()
plt.tight_layout()
plt.savefig('document/figures/comparison-yhat.png')

fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='all')
for i in range(3):
    axes[i].set_title(r'$SS_E =$ ' + f'{ss_error[i]:.2f}')
    axes[i].set_xlabel('x')
    axes[i].plot(x, y[i], marker='o', markersize=3, linestyle='None', alpha=0.7,
                 label=f'y{i+1}', color=color_list[i])
    axes[i].plot(x, y_hat[i], marker='o', markersize=3, lw='.5', alpha=0.7,
                 label=r'$\hat{y}$' + f'{i + 1}', color='#d62728')
    for j in range(n):
        axes[i].plot([x[j], x[j]], [y_hat[i, j], y[i, j]], lw='.5',
                     color=color_list[i])
    axes[i].legend()
plt.tight_layout()
plt.savefig('document/figures/comparison-sse.png')

fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='all')
for i in range(3):
    axes[i].set_title(r'$SS_R =$ ' + f'{ss_regression[i]:.2f}')
    axes[i].set_xlabel('x')
    axes[i].plot(x, y_bar[i], marker='o', markersize=3, lw='.5', alpha=0.7,
                 label=r'$\bar{y}$' + f'{i + 1}', color='black')
    axes[i].plot(x, y_hat[i], marker='o', markersize=3, lw='.5', alpha=0.7,
                 label=r'$\hat{y}$' + f'{i + 1}', color='#d62728')
    for j in range(n):
        axes[i].plot([x[j], x[j]], [y_hat[i, j], y_bar[i, j]], lw='.5',
                     color=color_list[i])
    axes[i].legend()
plt.tight_layout()
plt.savefig('document/figures/comparison-ssr.png')

fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='all')
for i in range(3):
    axes[i].set_title(r'$R^2 =$ ' + f'{r_square[i] * 100:.1f}%')
    axes[i].set_xlabel('x')
    axes[i].plot(x, y[i], marker='o', markersize=3, linestyle='None', alpha=0.7,
                 label=f'y{i + 1}', color=color_list[i])
    axes[i].plot(x, y_hat[i], marker='o', markersize=3, lw='.5', alpha=0.7,
                 label=r'$\hat{y}$' + f'{i + 1}', color='#d62728')
    axes[i].legend()
plt.tight_layout()
plt.savefig('document/figures/comparison-rsquared.png')
#
#
# fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=100, sharex='all', sharey='all')
# for i in range(3):
#     axes[i].set_xlabel('x')
#     axes[i].plot(x, y_bar[i], marker='o', markersize=2, linestyle='None',
#                  label=r'$\bar{y}$' + f'{i + 1}', color='black')
#     axes[i].plot(x, y_hat[i], marker='.', markersize=4, lw='.5',
#                  label=r'$\hat{y}$' + f'{i + 1}', color='#d62728')
#     axes[i].legend()
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=100, sharex='all', sharey='all')
# for i in range(3):
#     axes[i].set_xlabel('x')
#     axes[i].plot(x, y_bar[i], marker='o', markersize=2, linestyle='None',
#                  label=r'$\bar{y}$' + f'{i + 1}', color='black')
#     axes[i].plot(x, y_hat[i], marker='.', markersize=4, lw='.5',
#                  label=r'$\hat{y}$' + f'{i + 1}', color='#d62728')
#     for j in range(n):
#         axes[i].plot([x[j], x[j]], [y_bar[i, j], y_hat[i, j]], lw='.5',
#                      color=color_list[i])
#     axes[i].legend()
# plt.tight_layout()
# plt.show()

print(ss_total - ss_regression - ss_error)