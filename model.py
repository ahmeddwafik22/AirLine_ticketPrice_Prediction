import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import timeit
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

## data encoding ##
# hot encoding
# airlines = 8
# ch_code = 8
# encoding_type = 2
# encoding_source = 6
# encoding_destination = 6
hot_encoding_airLines = pd.get_dummies(data['airline'], drop_first=False)
hot_encoding_ch_code = pd.get_dummies(data['ch_code'], drop_first=False)
hot_encoding_stop = pd.get_dummies(data['stop'], drop_first=False)
hot_encoding_type = pd.get_dummies(data['type'], drop_first=False)
hot_encoding_source = pd.get_dummies(data['Source'], drop_first=False)  # I think we wont use it
hot_encoding_destination = pd.get_dummies(data['Destination'], drop_first=False)  # I think we wont use it
# label encoding
data.drop(['Year', 'date', 'route', 'airline', 'ch_code', 'type', 'Source', 'Destination', 'stop'], inplace=True,
          axis=1)
## split into training and test data

# feature extraction


y = data['price']
x = data
x.drop(['price'], inplace=True, axis=1)


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


# cols=('Source','Destination')
# x=Feature_Encoder(x,cols)

x = pd.concat(
    [x, hot_encoding_airLines, hot_encoding_ch_code, hot_encoding_type, hot_encoding_source,hot_encoding_destination,
     hot_encoding_stop], axis=1)
# dataa = pd.concat ([x,y],axis = 1)
print(x.info())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


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
    print('RMSE', (np.sqrt(mean_squared_error(y_test, prediction))))
    import pickle
    filename = 'model.sav'
    pickle.dump(model,open(filename,'wb'))
m(4, x_train, y_train, x_test, y_test)