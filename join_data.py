import pandas as pd
pd.set_option('display.max_columns', 15)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv') 

print(X_train.head())
print(X_train.tail())
print(y_train.head())
print(y_train.tail())

print(X_train['orientation_X'].max())
print(X_train['orientation_X'].min())

#df = pd.merge(X_train, y_train, on = ['series_id'])
df.to_csv('final_data.csv', index = False)