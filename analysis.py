import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 20)
train = pd.read_csv('X_train.csv')

#print(train.shape)
#print(train.head())

temp = pd.read_csv('y_train.csv')
df = pd.merge(train, temp, on = ['series_id'])
print(df.head())
print(df.shape)

X_temp = df.iloc[:, 3 : 13]

#extract feature columns
#X_temp = train.iloc[:, 3:]

#derive linear velocity from angular velocity
X_temp['linear_velocity_X'] = X_temp['angular_velocity_X'] * X_temp['orientation_X']
X_temp['linear_velocity_Y'] = X_temp['angular_velocity_Y'] * X_temp['orientation_Y']
X_temp['linear_velocity_Z'] = X_temp['angular_velocity_Z'] * X_temp['orientation_Z']

#derive centripetal accelaration from angular velocity
X_temp['centripetal_acceleration_X'] = (X_temp['angular_velocity_X'] ** 2) * X_temp['orientation_X']
X_temp['centripetal_acceleration_Y'] = (X_temp['angular_velocity_Y'] ** 2) * X_temp['orientation_Y']
X_temp['centripetal_acceleration_Z'] = (X_temp['angular_velocity_Z'] ** 2) * X_temp['orientation_Z']
X_temp['surface'] = df['surface']


X_temp.to_csv('measure_train.csv', index = False)
#get number of series
#n_series = X_temp.shape[0] // 128

#initialize an empty dataframe
#X = pd.DataFrame()

#construct a dataframe of mean values
#for i in range(0, n_series):
    #temp = X_temp.iloc[128 * i : 128 * (i + 1), :]  #select 128 rows at a time from all columns
    #row = temp.agg(['mean'])
    #X = X.append(row, ignore_index = True)     #append to dataframe





#X['surface'] = temp['surface']

#X.to_csv('mean_train.csv', index = False)




