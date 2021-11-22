import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

methods = ['mad', 'sklearn', 'lm in R']

weights = {'uscrime':{'mad': [-6.12462505e+03, 8.15759216e+01, 8.41220409e+01, 1.44802842e+02,
                              5.95149035e+01, 4.18895857e+01, 3.36511852e+02, 2.15073471e+01,
                              -4.87668107e-01, 1.77080359e+00, -4.13580438e+03, 1.45889085e+02,
                              4.50778685e-02, 5.53314381e+01, -4.47206067e+03, -1.75385587e-01],
                  'sklearn': [-5984.287604496813,
                              8.78301732e+01, -3.80345030e+00, 1.88324315e+02, 1.92804338e+02, -1.09421925e+02,
                              -6.63826145e+02, 1.74068555e+01,  -7.33008150e-01, 4.20446100e+00, -5.82710272e+03,
                              1.67799672e+02, 9.61662430e-02, 7.06720995e+01, -4.85526582e+03, -3.47901784e+00],
                  'lm in R': [-5.984288e+03,
                              8.783017e+01, -3.803450e+00, 1.883243e+02, 1.928043e+02, -1.094219e+02,
                              -6.638261e+02, 1.740686e+01, -7.330081e-01, 4.204461e+00, -5.827103e+03,
                              1.677997e+02, 9.616624e-02, 7.067210e+01, -4.855266e+03, -3.479018e+00]},
       'BostonHousing': {'mad': [2.64512690e+01, -8.45826166e-02, 3.00135994e-02, -4.90821196e-02, 2.97989544e+00,
                                 -1.13924369e+01, 4.28726696e+00, -4.95890526e-03, -1.17872725e+00, 1.35683804e-01,
                                 -4.01826493e-03, -8.79642253e-01, 9.75449063e-03, -5.00584661e-01],
                   'sklearn': [36.45948838508954, -1.08011358e-01, 4.64204584e-02, 2.05586264e-02, 2.68673382e+00,
                               -1.77666112e+01,  3.80986521e+00,  6.92224640e-04, -1.47556685e+00,
                               3.06049479e-01, -1.23345939e-02, -9.52747232e-01,  9.31168327e-03,
                               -5.24758378e-01],
                   'lm in R': [3.645949e+01, -1.080114e-01,  4.642046e-02,  2.055863e-02,  2.686734e+00, -1.776661e+01,
                               3.809865e+00,  6.922246e-04, -1.475567e+00,  3.060495e-01,
                               -1.233459e-02, -9.527472e-01,  9.311683e-03, -5.247584e-01]},
       'diamonds': {'mad': [1.4823e2, 2.1869, 2.1667e1, -4.46173e-1],
                   'sklearn': [148.33541, 2.18942, 21.6921, -0.45494],
                   'lm in R': [148.3354, 2.1894, 21.6922, -0.4549]},
       }


dataset = 'diamonds'
# dataset = 'uscrime'
# dataset = 'BostonHousing'

if dataset == 'uscrime':
    df = pd.read_csv('data/uscrime.txt', sep='\t')
elif dataset == 'BostonHousing':
    df = pd.read_csv('data/BostonHousing.txt', sep=',')
elif dataset == 'diamonds':
    df = pd.read_csv('data/diamonds.csv', sep=' ')

df = df.to_numpy()
x = df[:, 0:-1]
y = df[:, -1]

y_res = np.zeros((3, len(y)))
for i in range(3):
    y_res[i] = np.matmul(x, weights[dataset][methods[i]][1:]) + weights[dataset][methods[i]][0]

y1 = min(min(y), np.min(y_res)) - 0.05 * abs(min(min(y), np.min(y_res)))
y2 = max(max(y), np.max(y_res)) + 0.05 * abs(max(max(y), np.max(y_res)))

fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
ax.set_title(f'{dataset}')
ax.plot(y, y_res[1], marker='x', markersize=5, linestyle='None', color='#ff7f0e',
        label='sklearn')
ax.plot(y, y_res[2], marker='+', markersize=5, linestyle='None', color='black',
        label='lm in R')
ax.plot(y, y_res[0], marker='o', markersize=4, linestyle='None', color='#2ca02c', alpha=0.7,
        label='our MAD code')
ax.axline((np.mean(y), np.mean(y)), slope=1., color='red')
ax.set_ylabel('Predicted value')
ax.set_xlabel('True value')
ax.set_xlim([y1, y2])
ax.set_ylim([y1, y2])
ax.legend()
plt.tight_layout()
plt.savefig(f'document/figures/compare-{dataset}.png')


