# LinearRegression
### LinearRegression.py
A Multi Linear Regression (MLR) method is implemented in LinearRegression.py

Under the class **LinearRegression(x, y)**, the following functions are provided:
- *standard_transform(x)*: center and scale *x* data.
- *gradient_descent(x, y, lr)*: perform gradient descent on *x* and *y* using the provided learning rate *lr* to determine the weights.
- *tune_learning_rate()*: call gradient_descent with different learning rates and determine the optimal learning rate
- *fit(lr)*: train the model with provided *lr*. If *lr* is not specified, then tune_learning_rate is called to get an optimal learning rate to use in *fit*.
- *predict(x)*: predict y of the input test data *x*.
- *get_unscaled_weights()*: output the weights of the linear model.
- *significance_test()*: perform significance tests and then output F-statistic, P-value of the regression model, R-squared, adjusted R-squared, t-value and P-value of each weights.


### test_random_data.py
Unittest of *LinearRegression* using randomly generated data.

### test_datasets.py
Apply *LinearRegression* on existing data sets:
- *uscrime*: http://www.statsci.org/data/general/uscrime.html
- *BostonHousing*: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
- *diamonds*: https://rdrr.io/cran/yarrr/man/diamonds.html

