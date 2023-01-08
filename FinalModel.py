import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import timeit
import time as t
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import xgboost as xgb
# from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("airline-price-prediction.csv")
# edit date format
data['date'] = data['date'].str.replace('/', '-')
# edit price format
data['price'] = data['price'].str.replace(',', '')
# edit time_taken format
data['time_taken'] = data['time_taken'].str.replace('h ', ':')
data['time_taken'] = data['time_taken'].str.replace('m', '')
# split route into Source and Destination
temp = data['route'].str.split(',', expand=True)
data['Source'] = temp[0].str.split(':', expand=True)[1]
data['Destination'] = temp[1].str.split(':', expand=True)[1]
data['Destination'] = data['Destination'].str.replace('}', '')
data['Destination'] = data['Destination'].str.replace("'", '')
data['Source'] = data['Source'].str.replace("'", '')
# split date into day and month and year
data['Day'] = data['date'].str.split('-', expand=True)[0]
data['Month'] = data['date'].str.split('-', expand=True)[1]
data['Year'] = data['date'].str.split('-', expand=True)[2]
# convert dep_time to minutes
temp = data['dep_time'].str.split(':', expand=True)
data['dep_time'] = (temp[0].astype(float) * 60).astype(int) + temp[1].astype(int)
# convert time_taken to minutes
hours = data['time_taken'].str.split(":", expand=True)[0]
minutes = data['time_taken'].str.split(":", expand=True)[1]
minutes = minutes.apply(lambda x: '0' if x == "" else x)
data['time_taken'] = (hours.astype(float) * 60).astype(int) + minutes.astype(int)
# convert arrtime to minutes
hours = data['arr_time'].str.split(":", expand=True)[0]
minutes = data['arr_time'].str.split(":", expand=True)[1]
minutes = minutes.apply(lambda x: '0' if x == "" else x)
data['arr_time'] = (hours.astype(float) * 60).astype(int) + minutes.astype(int)
((data["arr_time"] == data["time_taken"] + data["dep_time"]) == False).sum()
# edit datatype to number columns
data['Day'] = data['Day'].astype(int)
data['Month'] = data['Month'].astype(int)
data['Year'] = data['Year'].astype(int)
data['price'] = data['price'].astype(int)
data['num_code'] = data['num_code'].astype(int)
data['stop'] = data.loc[:, 'stop'].apply(lambda x: '0' if x[0] == 'n' else x[0])
data.drop(['Year', 'date', 'route', 'arr_time', 'Month', 'Day'], inplace=True, axis=1)

# label encoding

# /cols = ['Source', 'Destination', 'airline', 'ch_code', 'stop', 'type']

cols = ['Source', 'Destination', 'airline', 'ch_code', 'stop', 'type']

y = data['price']
x = data.copy()
x.drop(['price', 'airline', 'ch_code', 'type', 'Source', 'Destination'], inplace=True, axis=1)

# print(pd.get_dummies(x,drop_first=True).columns)
np.save("regressionHot", pd.get_dummies(x, drop_first=True).columns)
x = pd.get_dummies(x, drop_first=True)


# encoder = OrdinalEncoder()
# encoder.fit(data[cols])
# data[cols] = encoder.transform(data[cols])
# np.save('regression.npy', encoder.categories_, allow_pickle=True)
# print(data.columns)
# y = data['price']
# x = data.copy()
# x.drop(['price'], inplace=True, axis=1)
# x.drop(['Year', 'date', 'route'], inplace=True, axis=1)


## Training
def m(d, x_train, y_train, x_test, y_test):
    model_poly_features = PolynomialFeatures(degree=d)
    X_train_poly_model = model_poly_features.fit_transform(x_train)
    model = linear_model.LinearRegression()
    model.fit(X_train_poly_model, y_train)
    prediction = model.predict(model_poly_features.fit_transform(x_test))
    print(r2_score(y_test, prediction))
    print("Model with polynomial degree hot encoding %d MSE %d" % (d, metrics.mean_squared_error(y_test, prediction)))
    print('MAE:', (mean_absolute_error(y_test, prediction)))
    print('RMSE', np.sqrt(mean_squared_error(y_test, prediction)))
    import pickle
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))


def saveModel(Name, model):
    filename = Name + '.sav'
    pickle.dump(model, open(filename, 'wb'))


def models():
    print("Regression Models")
    m(1, x_train, y_train, x_test, y_test)
    m(2, x_train, y_train, x_test, y_test)
    m(3, x_train, y_train, x_test, y_test)

    model = DecisionTreeRegressor(max_depth=200)

    print("Decision Tree Model : ")
    model.fit(x_train, y_train)
    p = model.predict(x_test)
    print(r2_score(y_test, p))
    print('RMSE for decision Tree ', (mean_squared_error(y_test, p)))
    print("Random Forest Model")
    model = RandomForestRegressor(n_estimators=10, max_depth=100)
    model.fit(x_train, y_train)
    p = model.predict(x_test)
    print(r2_score(y_test, p))


x = pd.concat([x, hot_encoding_ch_code, hot_encoding_type], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=120)

m(3, x_train, y_train, x_test, y_test)
model = DecisionTreeRegressor(max_depth=200)
model.fit(x_train, y_train)
p = model.predict(x_test)
print(r2_score(y_test, p))
print('RMSE', (mean_squared_error(y_test, p)))

saveModel("DecisionTreeRegressor", model)

print("RANDOM FOREST MODEL v1")
model = RandomForestRegressor(n_estimators=10, max_depth=100)
model.fit(x_train, y_train)
p = model.predict(x_test)
print(r2_score(y_test, p) * 100)

saveModel("RandomForestRegressorV1", model)

print("RANDOM FOREST MODEL v2")

model = RandomForestRegressor(n_estimators=50, max_depth=50)
model.fit(x_train, y_train)
p = model.predict(x_test)
print(r2_score(y_test, p) * 100)
model.fit(x_train, y_train)
p = model.predict(x_test)
print(r2_score(y_test, p) * 100)

saveModel("RandomForestRegressorV2", model)

from sklearn.tree import DecisionTreeRegressor

start = t.perf_counter()
model = DecisionTreeRegressor(max_depth=100)
model.fit(x_train, y_train)
p = model.predict(x_test)
print("model with decision tree " + str(r2_score(y_test, p)))
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")

saveModel("DecisionTreeRegressor", model)

import xgboost as xgb
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=50, max_depth=100)

start = t.perf_counter()
model.fit(x_train, y_train)
end = t.perf_counter()

p = model.predict(x_test)

saveModel("XGBRegressor", model)

print("Model with xgboost with num_estimators(50) and max_depth(100) MSE %f" % (metrics.mean_squared_error(y_test, p)))
print("accuracy " + str(r2_score(y_test, p) * 100))
print("time is: " + str(end - start) + "sec")
sns.jointplot(y_test, p)

model = XGBRegressor()

start = t.perf_counter()
model.fit(x_train, y_train)
end = t.perf_counter()

p = model.predict(x_test)

saveModel("XGBRegressor2", model)

print("Model with xgboost with  Random num_estimators and Random max_depth MSE %f" % (
    metrics.mean_squared_error(y_test, p)))
print("accuracy " + str(r2_score(y_test, p) * 100))
print("time is: " + str(end - start) + "sec")
