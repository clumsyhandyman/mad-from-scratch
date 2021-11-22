import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

np.random.seed(seed=42)


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

    learner = LinearRegression().fit(x, y)
    y_pred = learner.predict(x)

    print(learner.intercept_)
    print(learner.coef_)
    print(learner.score(x, y))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    ax.set_title(f'{dataset}')
    ax.plot(y, y_pred, marker='o', markersize=4, linestyle='None', color='#1f77b4')
    ax.axline((np.mean(y), np.mean(y)), slope=1., color='red')
    ax.set_ylabel('Predicted value')
    ax.set_xlabel('True value')

    plt.tight_layout()
    plt.show()



# test('uscrime')
test('BostonHousing')
# test('diamonds')

# test_learning_rate('BostonHousing')

