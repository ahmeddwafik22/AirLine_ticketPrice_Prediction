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

Modes = {}


def saveModel(Name, model):
    filename = Name + '.sav'
    pickle.dump(model, open(filename, 'wb'))


data = pd.read_csv("airline-price-prediction.csv")

d  = data.copy()
d.drop('price',axis = 1,inplace=True)

for i in d.columns:
    if d[i].dtype == object:
        Modes[i] = d[i].mode()[0]
    else:
        Modes[i] = d[i].mean()

np.save("modesRegg", Modes)

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

## data encoding ##
# hot encoding


data['price'] = np.where(data['price'] >= 120000, data['price'].median(), data['price'])

source = data['Source']
destination = data['Destination']

y = data['price']
x = data.copy()
x.drop(['price'], inplace=True, axis=1)
#
# x.drop(['Year', 'date', 'route', 'arr_time'], inplace=True, axis=1)
#
# x.drop(['airline', 'ch_code', 'type'], inplace=True, axis=1)
#
# x.drop(['Source', 'Destination'], inplace=True, axis=1)

# hot_encoding_airLines = pd.get_dummies(data['airline'], drop_first=False)
# hot_encoding_ch_code = pd.get_dummies(data['ch_code'], drop_first=False)
# # hot_encoding_stop = pd.get_dummies(data['stop'],drop_first=False)
# hot_encoding_type = pd.get_dummies(data['type'], drop_first=False)
x['stop'] = x['stop'].astype(int)
# x = pd.concat([x, hot_encoding_ch_code, hot_encoding_type], axis=1)
# np.save("regressionHot", pd.get_dummies(x, drop_first=True).columns)
np.save("regressionHot", pd.get_dummies(x, drop_first=True).columns)
x = pd.get_dummies(x, drop_first=True)

print(len(x.columns))


def m(d, x_train, y_train, x_test, y_test):
    model_poly_features = PolynomialFeatures(degree=d)
    X_train_poly_model = model_poly_features.fit_transform(x_train)
    model = linear_model.LinearRegression()

    model.fit(X_train_poly_model, y_train)

    prediction = model.predict(model_poly_features.fit_transform(x_test))

    saveModel("poly" + str(d), model)

    print("accuracy is", r2_score(y_test, prediction))
    print("Model with polynomial degree hot encoding %d MSE %f" % (d, metrics.mean_squared_error(y_test, prediction)))
    print('MAE:', (mean_absolute_error(y_test, prediction)))
    sns.jointplot(y_test, prediction)


# hot_encoding_source = pd.get_dummies(data['Source'], drop_first=False)  # I think we wont use it
# x = pd.concat([x, hot_encoding_source], axis=1)

# for i in source.unique():
#     x.rename(columns={i: i + "_source"}, inplace=True)
#
# # hot_encoding_destination = pd.get_dummies(data['Destination'], drop_first=False)
# # x = pd.concat([x, hot_encoding_destination], axis=1)
#
# for i in destination.unique():
#     x.rename(columns={i: i + "_destination"}, inplace=True)

# hot_encoding_source = pd.get_dummies(data['Source'], drop_first=False) # I think we wont use it
# hot_encoding_destination = pd.get_dummies(data['Destination'], drop_first=False) # I think we wont use it


# In[854]:


from sklearn.ensemble import ExtraTreesRegressor

selection = ExtraTreesRegressor()
selection.fit(x, y)
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(selection.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

# In[855]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=120)

# In[856]:


import time as t

start = t.perf_counter()
m(3, x_train, y_train, x_test, y_test)
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")

# In[858]:


start = t.perf_counter()
m(2, x_train, y_train, x_test, y_test)
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")

# In[859]:


start = t.perf_counter()
m(1, x_train, y_train, x_test, y_test)
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")

from sklearn.tree import DecisionTreeRegressor

start = t.perf_counter()
model = DecisionTreeRegressor(max_depth=100)
model.fit(x_train, y_train)
p = model.predict(x_test)
print("model with decision tree " + str(r2_score(y_test, p)))
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")
sns.jointplot(y_test, p)

saveModel("DecisionTreeRegressor", model)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, max_depth=100)
star = t.perf_counter()
model.fit(x_train, y_train)
end = t.perf_counter()
p = model.predict(x_test)

saveModel("RandomForestRegressor_small", model)

print("Model with random forest with num_estimators(10) and max_depth(100) MSE %f" % (
    metrics.mean_squared_error(y_test, p)))
print("accuracy " + str(r2_score(y_test, p) * 100))
print("time is: " + str(end - start) + "sec")
sns.jointplot(y_test, p)

# In[862]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50, max_depth=100)
star = t.perf_counter()
model.fit(x_train, y_train)
end = t.perf_counter()
p = model.predict(x_test)

saveModel("RandomForestRegressor_High", model)

print("Model with random forest with num_estimators(50) and max_depth(100) MSE %f" % (
    metrics.mean_squared_error(y_test, p)))
print("accuracy " + str(r2_score(y_test, p) * 100))
print("time is: " + str(end - start) + "sec")
sns.jointplot(y_test, p)

import xgboost as xgb
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=50, max_depth=100)

start = t.perf_counter()
model.fit(x_train, y_train)
end = t.perf_counter()

p = model.predict(x_test)

saveModel("XGBRegressor_minEstimator", model)

print("Model with xgboost with num_estimators(50) and max_depth(100) MSE %f" % (metrics.mean_squared_error(y_test, p)))
print("accuracy " + str(r2_score(y_test, p) * 100))
print("time is: " + str(end - start) + "sec")
sns.jointplot(y_test, p)

import xgboost as xgb
from xgboost import XGBRegressor

model = XGBRegressor()

start = t.perf_counter()
model.fit(x_train, y_train)
end = t.perf_counter()

p = model.predict(x_test)

saveModel("XGBRegressor_Default", model)

print("Model with xgboost with  Random num_estimators and Random max_depth MSE %f" % (
    metrics.mean_squared_error(y_test, p)))
print("accuracy " + str(r2_score(y_test, p) * 100))
print("time is: " + str(end - start) + "sec")
sns.jointplot(y_test, p)

from sklearn.preprocessing import StandardScaler

# In[869]:


xx = x.copy()
xx = StandardScaler().fit_transform(xx)
xx = pd.DataFrame(xx, columns=x.columns)

xx.head()

x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2, random_state=120)

start = t.perf_counter()

m(2, x_train, y_train, x_test, y_test)
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")
from sklearn.decomposition import PCA

pca = PCA(n_components=17)
comp = pca.fit_transform(x)

comp = pd.DataFrame(comp)

x_train, x_test, y_train, y_test = train_test_split(comp, y, test_size=0.2, random_state=120)

start = t.perf_counter()
m(2, x_train, y_train, x_test, y_test)
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")

start = t.perf_counter()
m(3, x_train, y_train, x_test, y_test)
end = t.perf_counter()
print("time is: " + str(end - start) + "sec")
