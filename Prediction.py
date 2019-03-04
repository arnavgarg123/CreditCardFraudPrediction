from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df1 = pd.read_csv('/home/arnav/WorkSpace/GitHub/CreditCardFraudPrediction/creditcard.csv')
df1.info()
df1.head()
df1.describe()
df1.isnull().sum()
df1.drop('Time', axis=1, inplace=True)
df1.drop('V1', axis=1, inplace=True)
df1.drop('V2', axis=1, inplace=True)
df1.drop('V3', axis=1, inplace=True)
df1.drop('V5', axis=1, inplace=True)
df1.drop('V8', axis=1, inplace=True)
df1.drop('V9', axis=1, inplace=True)
df1.drop('V10', axis=1, inplace=True)
df1.drop('V12', axis=1, inplace=True)
df1.drop('V14', axis=1, inplace=True)
df1.drop('V15', axis=1, inplace=True)
df1.drop('V16', axis=1, inplace=True)
df1.drop('V17', axis=1, inplace=True)
df1.drop('V18', axis=1, inplace=True)
df1.drop('V22', axis=1, inplace=True)
df1.drop('V23', axis=1, inplace=True)
df1.drop('V24', axis=1, inplace=True)
sns.heatmap(df1.corr(), annot=True)
sns.distplot(df1)
X = df1.drop(['Class'], axis=1)
y = df1['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1011)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.accuracy_score(y_pred, y_test) * 100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(metrics.accuracy_score(y_pred, y_test) * 100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
