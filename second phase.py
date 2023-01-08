import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
import json
import pickle
from sklearn import svm
from sklearn.svm import SVC

Modes = {}
data = pd.read_csv("airline-price-classification.csv")

lbl = LabelEncoder()
lbl.fit(list(data['TicketCategory'].values))
data['TicketCategory'] = lbl.transform(list(data['TicketCategory'].values))
d  = data.copy()
d.drop('TicketCategory',axis = 1,inplace=True)

for i in d.columns:
    if d[i].dtype == object:
        Modes[i] = d[i].mode()[0]
    else:
        Modes[i] = d[i].mean()

np.save("modes_class", Modes)

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

y = data['TicketCategory']
x = data.copy()
x.drop(['TicketCategory'], inplace=True, axis=1)
# print(pd.get_dummies(x,drop_first=True).columns)
np.save("hello", pd.get_dummies(x, drop_first=True).columns)
x = pd.get_dummies(x, drop_first=True)

# split to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=120)
# logistic regression
LG = LogisticRegression(random_state=0).fit(x_train, y_train)
logisticregression = LG.score(x_test, y_test)
print('Logistic Regresion accuracy : ', logisticregression * 100)
filename = 'Logistic_Regresion.sav'
pickle.dump(LG, open(filename, 'wb'))

# split to train and test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
filename = 'Decision_tree.sav'
pickle.dump(clf, open(filename, 'wb'))

# we create an instance of SVM and fit out data.
C = 1
# svc = svm.SVC(kernel='linear', C=C).fit(x, y)  # one vs one too long
lin_svc = svm.LinearSVC(C=C).fit(x, y)  # one vs all
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(x, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x, y)

filename = 'lin_svc.sav'

pickle.dump(lin_svc, open(filename, 'wb'))

filename = 'rbf_svc.sav'

pickle.dump(rbf_svc, open(filename, 'wb'))

filename = 'poly_svc.sav'

pickle.dump(poly_svc, open(filename, 'wb'))
