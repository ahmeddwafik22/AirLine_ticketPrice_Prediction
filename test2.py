
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
from sklearn.tree import DecisionTreeClassifier
import timeit
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("airline-test-samples.csv")
# edit date format
data['date'] = data['date'].str.replace('/', '-')
# edit price format

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

data['num_code'] = data['num_code'].astype(int)
data['stop'] = data.loc[:, 'stop'].apply(lambda x: '0' if x[0] == 'n' else x[0])
source = data['Source']
destination = data['Destination']


# CATEGORICAL ENCODING FUNCTION
def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


# CATEGORICAL ENCODING
cols = {'TicketCategory'}
data = Feature_Encoder(data, cols)

y = data['TicketCategory']
x = data.copy()
x.drop(['TicketCategory'], inplace=True, axis=1)
x.drop(['Year', 'date', 'route', 'arr_time'], inplace=True, axis=1)
x.drop(['airline', 'ch_code', 'type'], inplace=True, axis=1)
x.drop(['Source', 'Destination'], inplace=True, axis=1)
hot_encoding_airLines = pd.get_dummies(data['airline'], drop_first=False)
hot_encoding_ch_code = pd.get_dummies(data['ch_code'], drop_first=False)
# hot_encoding_stop = pd.get_dummies(data['stop'],drop_first=False)
hot_encoding_type = pd.get_dummies(data['type'], drop_first=False)
x = pd.concat([x, hot_encoding_ch_code, hot_encoding_type], axis=1)
hot_encoding_source = pd.get_dummies(data['Source'], drop_first=False)  # I think we wont use it
x = pd.concat([x, hot_encoding_source], axis=1)
for i in source.unique():
    x.rename(columns={i: i + "_source"}, inplace=True)
hot_encoding_destination = pd.get_dummies(data['Destination'], drop_first=False)
x = pd.concat([x, hot_encoding_destination], axis=1)
for i in destination.unique():
    x.rename(columns={i: i + "_destination"}, inplace=True)

    filename = "Decision_tree.sav"
    Decision_tree = pickle.load(open(filename, 'rb'))
    y_pred = Decision_tree.predict(x)
    print("Dession Tree Accuracy:", metrics.accuracy_score(y, y_pred) * 100)

    filename = "Logistic_Regresion.sav"
    Logistic_Regresion = pickle.load(open(filename, 'rb'))
    y_pred = Logistic_Regresion.predict(x)
    print("Logistic Regresion Accuracy:", metrics.accuracy_score(y, y_pred) * 100)

