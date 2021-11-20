import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class LinearRegression:
    def __init__(self, x, y, max_iteration=1000, tol=1e-3, verbose=False, plot=False):
        """
        :param x: x of training data
        :param y: y of training data
        :param max_iteration: maximum allowable number of iterations
        :param tol: tolerance to determine convergence
        :param verbose: whether to print out verbose
        :param plot: whether to output figures
        """
        self.max_iteration = max_iteration
        self.tol = tol
        self.verbose = verbose
        self.plot = plot
        self.x_raw = x
        self.mu = np.mean(x, axis=0)
        self.sigma = np.std(x, axis=0)
        self.x_train = self.standard_transform(x)
        self.y_train = y
        self.weights = None

    def standard_transform(self, x):
        """
        Scale and center data x
        :param x: input data
        :return: scaled_x: x after center and scale with a column of 1 inserted at beginning for intercept
        """
        scaled_x = (x - self.mu) / self.sigma
        scaled_x = np.insert(scaled_x, 0, np.ones(scaled_x.shape[0]), axis=1)
        return scaled_x

    def gradient_descend(self, x, y, lr=0.01):
        """
        this function is the gradient descent process of training data x and y.
        the weights(w) is optimized so that wx has the minimal MSE to y.
        :param x: attributes of training data. numpy.ndarray(n x d)
        :param y: outcome of training data. numpy.ndarray(n)
        :param lr: learning rate (alpha) of gradient descent
        :return: w: weights so that y_pred = wx with minimal MSE to y
        :return: mse_log: records of MSE of each iteration
        """
        if self.verbose is True:
            print(f'====== START gradient descent with learning rate of {lr}')
        # Get number of instances (n) and number of attributes (d)
        n = x.shape[0]
        d = x.shape[1]

        # Initialize weights and bias with zeros
        w = np.zeros(d)

        # Initialize list to store MSE
        mse_log = []

        # Loop for gradient descent
        for i in range(self.max_iteration):
            y_pred = np.matmul(x, w)
            mse = np.mean((y - y_pred) ** 2)
            if self.verbose is True:
                print(f'Iteration: {i} mse = {mse}')

            # Terminate if decrease of MSE is less than tol
            if len(mse_log) != 0 and 0 <= (mse_log[-1] - mse) / mse_log[-1] < self.tol:
                if self.verbose is True:
                    print('MSE improvement is less than tol. Terminate.')
                break
            elif len(mse_log) != 0 and mse_log[-1] < mse:
                if self.verbose is True:
                    print('MSE increases. Failed to converge.')
                break
            else:
                mse_log.append(mse)

            gradient_w = -2 * np.matmul((y - y_pred), x) / n
            w -= lr * gradient_w

        return w, mse_log

    def tune_learning_rate(self):
        """
        try gradient descend with a lr of 0.1, then try 0.01, 0.001...
        stop when MSE is 10 times larger than the previous lr
        :return: opt_lr: optimal learning rate (lr resulting the minimum MSE)
        """
        lr = 0.1
        w, mse_log = self.gradient_descend(self.x_train, self.y_train, lr=lr)
        if self.verbose is True:
            print(f'lr = {lr} mse = {mse_log[-1]}')
        lr_list = [lr]
        mse_list = [mse_log[-1]]
        mse_log_list = [mse_log]

        while lr > 1e-20:
            lr *= 0.1
            w, mse_log = self.gradient_descend(self.x_train, self.y_train, lr=lr)
            if self.verbose is True:
                print(f'lr = {lr} mse = {mse_log[-1]}')
            # if mse is 10 times larger than previous iteration, stop
            if mse_log[-1] > 10 * mse_list[-1]:
                break
            lr_list.append(lr)
            mse_list.append(mse_log[-1])
            mse_log_list.append(mse_log)

        opt_lr = lr_list[mse_list.index(min(mse_list))]

        legend_list = []
        for lr in lr_list:
            legend_list.append(f'{lr:.0e}')

        if self.plot is True:
            fig, axes = plt.subplots(2, 1, figsize=(4.5, 5), dpi=100)

            axes[0].set_title('Learning curves of different learning rates')
            for i in range(len(lr_list)):
                axes[0].plot(mse_log_list[i])
            axes[0].set_ylabel('MSE')
            axes[0].set_xlabel('Iteration')
            axes[0].legend(legend_list)

            axes[1].set_title('Converged MSE of different learning rates')
            axes[1].plot(lr_list, mse_list, marker='s', color='red')
            axes[1].plot([opt_lr], [min(mse_list)], marker='o', color='#2ca02c',
                         mfc='none', linestyle='none', markersize=12,
                         label=f'optimum learning rate = {opt_lr}')
            axes[1].set_xscale('log')
            axes[1].invert_xaxis()
            axes[1].set_xlabel('Learning rate')
            axes[1].set_ylabel('MSE')
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        return opt_lr

    def fit(self, lr=None):
        """
        this function is the training process of training data x and y.
        :param lr: learning rate.
        If using default value of None, then tune_learning_rate is called and the optimal learning rate is used as lr.
        :return: None
        """
        if lr is None:
            lr = self.tune_learning_rate()
        w, mse_log = self.gradient_descend(self.x_train, self.y_train, lr=lr)
        self.weights = w
        if self.plot is True:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), dpi=100)
            ax.set_title('Learning curve of linear regression')
            ax.plot(mse_log, marker='.', color='#ff7f0e')
            ax.set_ylabel('MSE')
            ax.set_xlabel('Iteration')
            ax.set_yscale('log')
            plt.tight_layout()
            plt.show()
        self.significance_test()

    def predict(self, x):
        """
        Predict y of given x
        :param x: input data
        :return: predicted y of input data
        """
        x_test = self.standard_transform(x)
        return np.matmul(x_test, self.weights)

    def get_unscaled_weights(self):
        """
        :return: weights for the unscaled x
        """
        unscaled_wights = self.weights[1:] / self.sigma
        unscaled_intercept = self.weights[0] - np.sum(self.weights[1:] * self.mu / self.sigma)
        unscaled_wights = np.insert(unscaled_wights, 0, unscaled_intercept)
        return unscaled_wights

    def significance_test(self):
        y = self.y_train
        y_pred = np.matmul(self.x_train, self.weights)

        n = self.x_train.shape[0]
        k = self.x_train.shape[1] - 1

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ms_total = ss_total / (n - 1)

        ss_regression = np.sum((y_pred - np.mean(y)) ** 2)
        ms_regression = ss_regression / k

        ss_error = np.sum((y - y_pred) ** 2)
        ms_error = ss_error / (n - k - 1)

        r_square = 1 - ss_error / ss_total
        adj_r_square = 1 - ms_error / ms_total

        f_ans = ms_regression / ms_error
        df1 = k
        df2 = n - k - 1
        p_ans = scipy.stats.f.sf(f_ans, df1, df2)

        f_ticks = np.linspace(scipy.stats.f.ppf(0.999, df1, df2),
                              scipy.stats.f.ppf(0.001, df1, df2), 1000)
        p1_ticks = scipy.stats.f.pdf(f_ticks, df1, df2)
        p2_ticks = scipy.stats.f.sf(f_ticks, df1, df2)

        f_5 = np.interp(0.05, p2_ticks, f_ticks)
        f_10 = np.interp(0.10, p2_ticks, f_ticks)

        if np.min(f_ticks) <= f_ans <= np.max(f_ticks):
            p_string = f'{p_ans:.3f}'
        elif f_ans > np.max(f_ticks):
            p_string = '< 0.001'
        else:
            p_string = '> 0.999'

        x = self.x_raw
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        mat_c = np.linalg.inv(np.matmul(np.transpose(x), x)) * ms_error

        se = np.sqrt(np.diagonal(mat_c))
        w = self.get_unscaled_weights()
        t0 = w / se
        p_weights = np.zeros(k + 1)
        p_weights[t0 >= 0] = 2 * scipy.stats.t.sf(t0[t0 >= 0], df2)
        p_weights[t0 < 0] = 2 * scipy.stats.t.cdf(t0[t0 < 0], df2)

        print('\n')
        print(f'-----------------------------------------------------------------------------------------')
        print(f'Source of  \t|\tDegrees of\t|\t Sum of   \t|\t  Mean    \t|\t    F     \t|\t  P')
        print(f'Variation  \t|\t Freedom  \t|\t Squares  \t|\t Squares  \t|\tStatistic \t|\tValue')
        print(f'-----------------------------------------------------------------------------------------')
        print(f'Regression \t|\t{k:10}\t|\t{ss_regression:10.4e}\t|\t{ms_regression:10.4e}\t|\t{f_ans:10.4e}'
              f'\t|\t{p_string}')
        print(f'Error      \t|\t{n - k - 1:10}\t|\t{ss_error:10.4e}\t|\t{ms_error:10.4e}\t|\t')
        print(f'Total      \t|\t{n - 1:10}\t|\t{ss_total:10.4e}\t|\t{ms_total:10.4e}\t|\t')
        print(f'-----------------------------------------------------------------------------------------')
        print(f'R-squared = {r_square * 100:.2f}% \t\t adjusted R-squared = {adj_r_square * 100:.2f}%')
        print(f'------------------------------------------------------------------------------')
        print(f'Term       \t|\tCoefficient\t|\t Standard \t|\t    t     \t|\t  P')
        print(f'           \t|\t           \t|\t  Error   \t|\t  Value   \t|\tValue')
        print(f'------------------------------------------------------------------------------')
        print(f'(Intercept)\t|\t{w[0]:10.4e}\t|\t{se[0]:10.4e}\t|\t{t0[0]:10.4e}\t|\t{p_weights[0]:10.4e}')
        for j in range(k):
            print(f'X{str(j+1):10}\t|\t{w[j+1]:10.4e}\t|\t{se[j+1]:10.4e}\t|\t{t0[j+1]:10.4e}\t|\t{p_weights[j+1]:10.4e}')
        print(f'------------------------------------------------------------------------------')

        if self.plot is True:
            fig, axes = plt.subplots(2, 1, sharex='all', figsize=(4, 5), dpi=100)
            axes[0].set_title(f'F distribution, df1 = {df1}, df2 = {df2}')
            axes[0].plot(f_ticks, p1_ticks, 'b', label='Probability density function')
            axes[1].plot(f_ticks, p2_ticks, color='#2ca02c', alpha=0.7, lw=2, label='Survival function (p-value)')
            axes[1].set_xlabel('F-statistic')
            axes[1].plot([0, f_10], [0.1, 0.1], color='#ff7f0e', label=f'F = {f_10:.3f}, p = {0.1:.2f}')
            axes[1].plot([f_10, f_10], [0, 0.1], color='#ff7f0e')
            axes[1].plot([0, f_5], [0.05, 0.05], color='red', label=f'F = {f_5:.3f}, p = {0.05:.2f}')
            axes[1].plot([f_5, f_5], [0, 0.05], color='red')
            if np.min(f_ticks) <= f_ans <= np.max(f_ticks):
                axes[1].plot([0, f_ans], [p_ans, p_ans], color='#1f77b4', linestyle='dashed',
                             label=f'F = {f_ans:.3f}, p = {p_string}')
                axes[1].plot([f_ans, f_ans], [0, p_ans], color='#1f77b4', linestyle='dashed')
                axes[1].set_title(f'F = {f_ans:.3f}, p = {p_string}')
            else:
                axes[1].set_title(f'F = {f_ans:.3f}, p {p_string}')
            for i in range(2):
                axes[i].axhline(color='black', alpha=0.7)
                axes[i].axvline(color='black', alpha=0.7)
                axes[i].legend()
            plt.tight_layout()
            plt.show()




