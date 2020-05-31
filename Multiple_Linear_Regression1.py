import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv('Train.csv')
X_test = pd.read_csv('Test.csv')
X_train = X.iloc[:,:-1].values
y_train = X.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X_opt = np.array(X_train[:, [0, 1, 2, 3, 4]], dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
print(regressor_OLS.summary())

"""SINCE THE P-VALUES FOR ALL VARIABLES ARE ZERO,
HENCE ALL INDEPENDENT VARIABLES ARE SIGNIFICANT,
THEREFORE WE WILL NOT ELIMINATE ANY OF THE VARIABLES."""