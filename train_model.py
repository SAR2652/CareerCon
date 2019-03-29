import pandas as pd
pd.set_option('display.max_columns', 20)
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#X = pd.read_csv('final_train.csv')
X = pd.read_csv('measure_train.csv')
print(X.head())

le = LabelEncoder()
y = le.fit_transform(X['surface'])

X.drop(['surface'], axis = 1, inplace = True)

mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)

logreg = LogisticRegression(solver = 'lbfgs')
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
ada = AdaBoostClassifier()
xgb = XGBClassifier()

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

#logreg.fit(X_train, y_train)
#dt.fit(X_train, y_train)
#knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
#ada.fit(X_train, y_train)
#xgb.fit(X_train, y_train)

#y_pred_logreg = logreg.predict(X_val)
#y_pred_dt = dt.predict(X_val)
y_pred_rf = rf.predict(X_val)
#y_pred_ada = ada.predict(X_val)
#y_pred_knn = knn.predict(X_val)
#y_pred_xgb = xgb.predict(X_val)

#print("Accuracy :")
#print("Logistic Regression : {}".format(accuracy_score(y_val, y_pred_logreg)))
#print("Decision Tree : {}".format(accuracy_score(y_val, y_pred_dt)))
print("Random Forest : {}".format(accuracy_score(y_val, y_pred_rf)))
#print("Ada Boost : {}".format(accuracy_score(y_val, y_pred_ada)))
#print("K Nearest Neighbors : {}".format(accuracy_score(y_val, y_pred_knn)))
#print("Gradient Boosted Trees : {}".format(accuracy_score(y_val, y_pred_xgb)))

pickle.dump(le, open('labels.sav', 'wb'))
pickle.dump(rf, open('model.sav', 'wb'))
pickle.dump(mms, open('scaler.sav', 'wb'))
