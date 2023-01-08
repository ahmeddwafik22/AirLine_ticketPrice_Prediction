import csv

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

from sklearn import tree
import pickle
from sklearn.preprocessing import OrdinalEncoder


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


# CATEGORICAL ENCODING


Modes = {}
tempDic = np.load("modes_class.npy", allow_pickle=True).reshape(1, 1)

if input("Enter 0 to run second phase test \nEnter 1 to run second phase \n :)  ") == "0":

    Modes = dict(enumerate(tempDic.flatten(), 1))

    data = pd.read_csv("airline-tas-classification-test.csv")
    lbl = LabelEncoder()
    lbl.fit(list(data['TicketCategory'].values))
    yy = lbl.transform(list(data['TicketCategory'].values))
    print(yy)
    print(Modes[1]['date'])
    # Fill the missing first
    for i in data.columns:
        if i == "TicketCategory":
            continue
        data[i].fillna(Modes[1][i], inplace=True)

    # edit date format
    data['date'] = data['date'].str.replace('/', '-')
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
    data.drop(['Year', 'date', 'route', 'arr_time', 'Month', 'Day'], inplace=True, axis=1)

    cols = ['Source', 'Destination', 'airline', 'ch_code', 'stop', 'type', 'TicketCategory']

    p = np.load("hello.npy", allow_pickle=True)

    print(p)

    count = 0

    # check categories

    #    y = data['TicketCategory']
    x = data.copy()
    x.drop(['TicketCategory'], inplace=True, axis=1)

    # for i in p:
    #      x[i] = x.apply((lambda x: 0))
    COLMS = pd.get_dummies(x, drop_first=True).columns
    x = pd.get_dummies(x, drop_first=True)

    cols = ['Source', 'Destination', 'airline', 'ch_code', 'stop', 'type', 'TicketCategory']

    for i in COLMS:
        if i not in p:
            x.drop(i, inplace=True)
    for i in p:
        if i not in x:
            x[i] = 0

    filename = "Decision_tree.sav"
    Decision_tree = pickle.load(open(filename, 'rb'))
    y_pred = Decision_tree.predict(x)
    # print("Dession Tree Accuracy:", metrics.accuracy_score(y, y_pred) * 100)

    filename = "Logistic_Regresion.sav"
    Logistic_Regresion = pickle.load(open(filename, 'rb'))
    y_pred = Logistic_Regresion.predict(x)
    # print("Logistic Regresion Accuracy:", metrics.accuracy_score(y, y_pred) * 100)

    models = ["Decision_tree.sav", "Logistic_Regresion.sav"]
    for filename in models:
        model = pickle.load(open(filename, 'rb'))
        y_pred = model.predict(x)
        print(yy)
        print(y_pred)
        # print(filename + " MSE :" + str(mean_squared_error(y, y_pred)))
        print(filename + " R2 Score :" + str(accuracy_score(yy, y_pred)))
        print("\n predicted values" + str(y_pred) + "\n actual values" + str(yy))





else:
    data = pd.read_csv("airline-tas-regression-test.csv")

    tempDic = np.load("modesRegg.npy", allow_pickle=True).reshape(1, -1)
    Modes = dict(enumerate(tempDic.flatten(), 1))

    # data = pd.read_csv("airline-test-samples.csv")

    # print(Modes[1]['date'])
    # Fill the missing first
    for i in data.columns:
        if (i == "price"):
            continue
        data[i].fillna(Modes[1][i], inplace=True)

    # edit date format
    print(data.head())
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

    # data['price'] = data['price'].fillna(data['price'].mean())
    # label encoding

    cols = ['Source', 'Destination', 'airline', 'ch_code', 'stop', 'type']

    encoder = np.load('regressionHot.npy', allow_pickle=True)

    # count = 0

    # for i in cols:
    #     data[i] = data[i].apply(
    #         lambda x: x if encoder.categories_.__contains__(x) else encoder.categories_.tolist()[count][0])
    #     count += 1

    # data[cols] = encoder.transform(data[cols])
    #
    # # data[cols] = encoder.transform(data[cols])
    # np.save('regression.npy', encoder.categories_, allow_pickle=True)
    # print(data.columns)
    y = data['price']

    x = data.copy()
    x['stop'] = x['stop'].astype(int)

    x.drop(['price'], inplace=True, axis=1)
    # x.drop(['Year', 'date', 'route', 'airline', 'ch_code', 'type', 'Source', 'Destination'], inplace=True, axis=1)

    COLMS = pd.get_dummies(x, drop_first=True).columns

    x = pd.get_dummies(x, drop_first=True)

    for i in COLMS:
        if i not in encoder:
            x.drop(i, inplace=True)

    for i in encoder:
        if i not in x:
            x[i] = 0

    print(len(x.columns))

    count = 1
    models = ["poly1.sav", "poly2.sav", "poly3.sav", "XGBRegressor_minEstimator.sav", "XGBRegressor_Default.sav",
              "DecisionTreeRegressor.sav", "RandomForestRegressor_High.sav", "RandomForestRegressor_small.sav"]
    for filename in models:
        model = pickle.load(open(filename, 'rb'))
        if filename.__contains__("poly"):
            model_2_poly_features = PolynomialFeatures(degree=count)
            count += 1
            # transforms the existing features to higher degree features.
            X_train_poly_model_2 = model_2_poly_features.fit_transform(x)
            y_pred = model.predict(X_train_poly_model_2)
            # print(filename + " MSE :" + str(mean_squared_error(y, y_pred)))
            print(filename + " R2 Score :" + str(r2_score(y, y_pred)))
            print("\n predicted values" + str(y_pred) + "\n actual values" + str(y))

        else:

            y_pred = model.predict(x)
            print(filename + " MSE :" + str(mean_squared_error(y, y_pred)))
            print(filename + " R2 Score :" + str(r2_score(y, y_pred)))
            print("\n predicted values" + str(y_pred) + "\n actual values" + str(y))
