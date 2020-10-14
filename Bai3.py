import pandas as pd
import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
import tensorflow as tf
import keras


start = datetime.now()
df = pd.read_csv('./data/train.csv')
df1 = pd.read_csv('./data/test.csv')
ids = df1["Id"]
X_test1 = df1[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'TotRmsAbvGrd', 'Fireplaces','MasVnrArea', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF']]
X_test1=X_test1.fillna(0)
X=df[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'TotRmsAbvGrd', 'Fireplaces','MasVnrArea', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF']]
X=X.fillna(0)
Y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

mat_x_train = np.matrix(X_train)
mat_y_train = np.matrix(y_train).reshape((np.shape(y_train)[0], 1))

mat_x_test = np.matrix(X_test)
mat_y_test = np.matrix(y_test).reshape((np.shape(y_test)[0], 1))
mat_test1=np.matrix(X_test1)

prepo_x_train=MinMaxScaler()
prepo_x_train.fit(mat_x_train)

prepo_x_test=MinMaxScaler()
prepo_x_test.fit(mat_x_test)

prepo_y_train=MinMaxScaler()
prepo_y_train.fit(mat_y_train)

prepo_y_test=MinMaxScaler()
prepo_y_test.fit(mat_y_test)

X_tr = prepo_x_train.transform(mat_x_train)
X_test = prepo_x_test.transform(mat_x_test)
y_train = prepo_y_train.transform(mat_y_train)
y_test = prepo_y_test.transform(mat_y_test)
mat_test11 = prepo_x_train.transform(mat_test1)

#linearRegression
cfl1 = linear_model.LinearRegression()
cfl1.fit(X_tr, y_train)
y_pred = cfl1.predict(mat_test11)

y_pred1 = prepo_y_train.inverse_transform(y_pred)

y_pred = pd.DataFrame(y_pred1)
y_pred["Id"] = ids
y_pred = y_pred.rename(columns={0: "SalePrice"})
y_pred = y_pred[["Id","SalePrice"]]
y_pred.to_csv("linearRegression.csv", index=False)
#RidgeRegression
alpha=10
rm = linear_model.Ridge(alpha=alpha)
ridge_model = rm.fit(X_tr, y_train)
y_pred = ridge_model.predict(mat_test11)
y_pred1 = prepo_y_train.inverse_transform(y_pred)

y_pred = pd.DataFrame(y_pred1)
y_pred["Id"] = ids
y_pred = y_pred.rename(columns={0: "SalePrice"})
y_pred = y_pred[["Id","SalePrice"]]
y_pred.to_csv("RidgelinearRegression.csv", index=False)

#XGboot:
regr = xgb.XGBRegressor(silent=False, random_state=15)
parameters = {'learning_rate':[0.001,0.01,0.1],'n_estimators':[100,300,500,700],'max_depth':[1,2,3]} 

#cross validation 
clf = GridSearchCV(regr,
                    param_grid = parameters,
                    scoring="neg_mean_squared_error",
                    cv = 5,
                    n_jobs = -1,
                    verbose = 1)
result = clf.fit(X_tr, y_train)
y_pred = result.predict(mat_test11)
y_pred = np.reshape(y_pred, (np.shape(y_pred)[0], 1))
y_pred1 = prepo_y_train.inverse_transform(y_pred)
y_pred = pd.DataFrame(y_pred1)
y_pred["Id"] = ids
y_pred = y_pred.rename(columns={0: "SalePrice"})
y_pred = y_pred[["Id","SalePrice"]]
y_pred.to_csv("XGboot.csv", index=False)
#Ensemble learning:

regr1 = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=0).fit(X_tr, y_train)
y_pred = regr1.predict(mat_test11)
y_pred = np.reshape(y_pred, (np.shape(y_pred)[0], 1))
y_pred1 = prepo_y_train.inverse_transform(y_pred)
y_pred = pd.DataFrame(y_pred1)
y_pred["Id"] = ids
y_pred = y_pred.rename(columns={0: "SalePrice"})
y_pred = y_pred[["Id","SalePrice"]]
y_pred.to_csv("Bagging.csv", index=False)
#ANN:
model1=tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu', input_shape=np.shape(X_tr[0])),
                            tf.keras.layers.Dropout(0.2),
                            #tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dense(32, activation='elu'),
                            tf.keras.layers.Dropout(0.2),
                            #tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dense(48, activation='tanh'),
                            tf.keras.layers.Dropout(0.2),
                           #tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dense(1, activation='sigmoid')])

model1.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.mse, metrics=['mse'])
model1.fit(X_tr, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=1)
y_pred = model1.predict(mat_test11)
y_pred1 = prepo_y_train.inverse_transform(y_pred)
y_pred = pd.DataFrame(y_pred1)
y_pred["Id"] = ids
y_pred = y_pred.rename(columns={0: "SalePrice"})
y_pred = y_pred[["Id","SalePrice"]]
y_pred.to_csv("ANN.csv", index=False)

