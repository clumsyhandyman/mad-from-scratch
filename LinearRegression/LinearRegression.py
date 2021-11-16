import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x, y, max_iteration=1000, tol=1e-3):
        """
        :param x: x of training data
        :param y: y of training data
        :param max_iteration: maximum allowable number of iterations
        :param tol: tolerance to determine convergence
        """
        self.max_iteration = max_iteration
        self.tol = tol
        self.mu = np.mean(x, axis=0)
        self.sigma = np.std(x, axis=0)
        self.x_train = self.standard_transform(x)
        self.y_train = y
        self.weights = None

    def standard_transform(self, x):
        """
        Scale and center data x
        :param x: input data
        :return: scaled_x: x after center and scale and then insert a column of 1 for intercepts
        """
        scaled_x = (x - self.mu) / self.sigma
        scaled_x = np.insert(scaled_x, 0, np.ones(scaled_x.shape[0]), axis=1)
        return scaled_x

    def gradient_descend(self, x, y, learning_rate=0.01, verbose=True):
        """
        this function is the gradient descent process of training data x and y.
        the weights(w) is optimized so that wx has the minimal MSE to y.
        :param x: attributes of training data. numpy.ndarray(n x d)
        :param y: outcome of training data. numpy.ndarray(n)
        :param learning_rate: learning rate (alpha) of gradient descent
        :param verbose: print verbose outputs during iterations
        :return: w: weights so that y_pred = wx with minimal MSE to y
        :return: mse_log: records of MSE of each iteration
        """
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
            if verbose is True:
                print(f'Iteration: {i} ----------------')
                print(f'mse = {mse}')

            # Terminate if decrease of MSE is less than tol
            if len(mse_log) != 0 and 0 <= (mse_log[-1] - mse) / mse_log[-1] < self.tol:
                if verbose is True:
                    print('MSE improvement is less than tol. Terminate.')
                break
            elif len(mse_log) != 0 and mse_log[-1] < mse:
                if verbose is True:
                    print('MSE increases. Failed to converge.')
                break
            else:
                mse_log.append(mse)

            gradient_w = -2 * np.matmul((y - y_pred), x) / n
            w -= learning_rate * gradient_w

        return w, mse_log

    def tune_learning_rate(self, plot=False):
        lr = 0.1
        w, mse_log = self.gradient_descend(self.x_train, self.y_train, learning_rate=lr, verbose=False)
        print(f'lr = {lr} mse = {mse_log[-1]}')
        lr_list = [lr]
        mse_list = [mse_log[-1]]
        mse_log_list = [mse_log]

        while lr > 1e-20:
            lr *= 0.1
            w, mse_log = self.gradient_descend(self.x_train, self.y_train, learning_rate=lr, verbose=False)
            print(f'lr = {lr} mse = {mse_log[-1]}')
            if mse_log[-1] > 10 * mse_list[-1]:
                break
            lr_list.append(lr)
            mse_list.append(mse_log[-1])
            mse_log_list.append(mse_log)

        opt_lr = lr_list[mse_list.index(min(mse_list))]

        legend_list = []
        for lr in lr_list:
            legend_list.append(np.format_float_scientific(lr, precision=1, trim='0'))

        if plot is True:
            fig, axes = plt.subplots(2, 1, figsize=(6, 8), dpi=100)

            axes[0].set_title('Learning curve of different learning rate')
            for i in range(len(lr_list)):
                axes[0].plot(mse_log_list[i])
            axes[0].set_ylabel('MSE')
            axes[0].set_xlabel('Iteration')
            axes[0].legend(legend_list)

            axes[1].set_title('Converged MSE of different learning rate')
            axes[1].plot(lr_list, mse_list, marker='s', color='red')
            axes[1].set_xscale('log')
            axes[1].invert_xaxis()
            axes[1].set_xlabel('Learning rate')
            axes[1].set_ylabel('MSE')
            axes[1].legend(['optimum learning rate =' + str(opt_lr)])

            plt.tight_layout()
            plt.show()

        return opt_lr

    def fit(self, lr=None, plot=False):
        """
        this function is the training process of training data x and y.
        :param lr: learning rate.
        If using default value of None, then tune_learning_rate is conducted and the optimal learning rate is used as lr.
        :param plot: plot learning curve (MSE versus iteration)
        :return: None
        """
        if lr is None:
            lr = self.tune_learning_rate(plot=plot)
        w, mse_log = self.gradient_descend(self.x_train, self.y_train, learning_rate=lr, verbose=False)
        self.weights = w
        if plot is True:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), dpi=100)
            ax.set_title('Learning curve of linear regression')
            ax.plot(mse_log, marker='.', color='#ff7f0e')
            ax.set_ylabel('MSE')
            ax.set_xlabel('Iteration')
            ax.set_yscale('log')
            plt.tight_layout()
            plt.show()

    # def summary(self):
    #     """
    #     this function calculates the performance of the model
    #     and outputs MSE, R-squared, and p-values
    #     :return:
    #     """
    #     x = self.x_train
    #     n = x.shape[0]
    #     x = np.insert(x, 0, np.ones(n), axis=1)
    #     p = x.shape[1]
    #
    #     y = self.y_train
    #     y_pred = np.matmul(x, self.weights)
    #
    #     tss = np.sum((y - np.mean(y)) ** 2)
    #     rss = np.sum((y - y_pred) ** 2)
    #     mse = rss / (n - p)
    #     rsq = 1 - rss / tss
    #     rsq_adjusted = 1 - (n - 1)/(n - p - 1) * rss / tss
    #     print(f'R-square = {rsq}')
    #     print(f'adjusted R-square = {rsq_adjusted}')
    #
    #     if n > 2:
    #         mat_t = np.matmul(np.transpose(x), x)
    #         # print(mat_t)
    #         mat_t = np.linalg.inv(mat_t)
    #         # print(mat_t)
    #         se = np.sqrt(mat_t.diagonal() * mse)
    #         # print('se = ', se)
    #         t = self.weights / se
    #         # print('t =', t)
    #         for i in range(len(t)):
    #             s = np.random.standard_t(n, size=100000)
    #             p = np.sum(s < t[i]) / float(len(s))
    #             print(f'---- {i} -----------')
    #             print('coefficient = ', self.weights[i])
    #             print('p-value =', 2 * min(p, 1 - p))

    def predict(self, x):
        """
        Predict y of given x
        :param x: input data
        :return: predicted y of input data
        """
        x_test = self.standard_transform(x)
        return np.matmul(x_test, self.weights)

