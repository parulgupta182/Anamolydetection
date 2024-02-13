import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Definitions
w_size = 120
n_features = 3

# Initialization
data = pd.read_csv("Features2.csv")
first_column = data.columns[0]
data = data.drop([first_column], axis=1)
scaler = StandardScaler()

"""
Sampling and preprocessing (each sample / window is normalized individually!)
"""
X = []
y = []

for i in range(len(data)-w_size):

    # Predictors: last "w_size" samples
    pred = data[i:i+w_size]
    pred_N = scaler.fit_transform(pred)
    #print(pred_N)

    # Targets: next sample
    targ = data[i+w_size:i+w_size+1]
    targ_N = scaler.transform(targ)[0]
    #print(targ_N)

    # Collect normalized predictors and targets
    y.append(targ_N)
    X.append(np.ndarray.flatten(pred_N))

# Reconvert to arrays
X = np.asarray(X)
y = np.asarray(y)

"""
Split into training / validation data
"""
#ind_train, ind_val = train_test_split(np.arange(0, len(X), 1), train_size=0.8)
#X_train = X[ind_train]
#X_val = X[ind_val]
#y_train = y[ind_train]
#y_val = y[ind_val]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

"""
Multiple linear regression model for all features
"""
# Initialization

MLR = LinearRegression()

# Fit MLR model and get model coefficients
MLR.fit(X_train, y_train)
coefficient = MLR.coef_
intercept = MLR.intercept_


# Get model predictions
y_train_pred = MLR.predict(X_train)
y_val_pred = MLR.predict(X_val)
y_test_pred= MLR.predict(X_test)

# Get model performance
mse_train = np.sqrt(np.mean(np.square(y_train - y_train_pred)))
mse_val = np.sqrt(np.mean(np.square(y_val - y_val_pred)))

mse_test = np.sqrt(np.mean(np.square(y_test - y_test_pred)))


print(mse_val, mse_train, mse_test)

#print('Variance score: {}'.format(MLR.score(y_test, y_test_pred)))

## plotting for residual errors

plt.style.use('fivethirtyeight')
plt.scatter(MLR.predict(X_train), MLR.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

plt.scatter(MLR.predict(X_test), MLR.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

## plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=2, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()

"""
Seperate linear regression model for each feature
"""
# Initialization
LR = {}
LR['models'] = []
LR['y_train_hat'] = []
LR['y_val_hat'] = []
LR['mse_train'] = []
LR['mse_val'] = []

# Fit MLR model and get model coefficients
for i in range(n_features):
    LR['models'].append(LinearRegression().fit(X_train[:, i*w_size:(i+1)*w_size], y_train[:, i:i+1]))
    LR['W'] = [model.coef_ for model in LR['models']]
    LR['b'] = [model.intercept_ for model in LR['models']]

# Get model predictions
for i in range(n_features):
    LR['y_train_hat'].append(LR['models'][i].predict(X_train[:, i*w_size:(i+1)*w_size]))
    LR['y_val_hat'].append(LR['models'][i].predict(X_val[:, i*w_size:(i+1)*w_size]))

# Get model performance
for i in range(n_features):
    LR['mse_train'].append(np.sqrt(np.mean(np.square(y_train[:, i:i+1] - LR['y_train_hat'][i]))))
    LR['mse_val'].append(np.sqrt(np.mean(np.square(y_val[:, i:i+1] - LR['y_val_hat'][i]))))

print(LR['mse_train'], LR['mse_val'])




