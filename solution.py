import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', 15)

test = pd.read_csv('X_test.csv')
print(test.head())

#extract feature columns
X_temp = test.iloc[:, 3:]

#derive linear velocity from angular velocity
X_temp['linear_velocity_X'] = X_temp['angular_velocity_X'] * X_temp['orientation_X']
X_temp['linear_velocity_Y'] = X_temp['angular_velocity_Y'] * X_temp['orientation_Y']
X_temp['linear_velocity_Z'] = X_temp['angular_velocity_Z'] * X_temp['orientation_Z']

#derive centripetal accelaration from angular velocity
X_temp['centripetal_acceleration_X'] = (X_temp['angular_velocity_X'] ** 2) * X_temp['orientation_X']
X_temp['centripetal_acceleration_Y'] = (X_temp['angular_velocity_Y'] ** 2) * X_temp['orientation_Y']
X_temp['centripetal_acceleration_Z'] = (X_temp['angular_velocity_Z'] ** 2) * X_temp['orientation_Z']

print(X_temp.columns)
#get number of series
n_series = X_temp.shape[0] // 128

#initialize an empty dataframe
X_test = pd.DataFrame()

#construct a dataframe of mean values
for i in range(0, n_series):
    temp = X_temp.iloc[128 * i : 128 * (i + 1), :]  #select 128 rows at a time from all columns
    row = temp.agg(['mean'])
    X_test = X_test.append(row, ignore_index = True)     #append to dataframe

le = pickle.load(open('labels.sav', 'rb'))
rf = pickle.load(open('model.sav', 'rb'))
mms = pickle.load(open('scaler.sav', 'rb'))

X_test = mms.fit_transform(X_test)
y_pred = rf.predict(X_test)

y_test = le.inverse_transform(y_pred)

df = pd.DataFrame()
df['series_id'] = np.array([x for x in range(0, n_series)]).T
df['surface'] = y_test.T

df.to_csv('solution.csv', index = False)