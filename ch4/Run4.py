import pandas as pd
from io import StringIO
import numpy as np

# csv_data = '''A,B,C,D
# 1.0,2.0,3.0,4.0
# 5.0,6.0,,8.0
# 0.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
# df = pd.read_csv(StringIO(csv_data))
# print(df)

# print(df.isnull().sum())
# print("dropna:\n", df.dropna())
# print("dropna axis:\n", df.dropna(axis=1))
# print("dropna how:\n", df.dropna(how='all'))
# print("dropna thresh:\n", df.dropna(thresh=4))
# print("dropna subset:\n", df.dropna(subset=['C']))

# from sklearn.preprocessing import Imputer
# imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imr = imr.fit(df)
# imputed_data = imr.transform(df.values)
# print(imputed_data)

# df = pd.DataFrame([
# ['green', 'M', 10.1, 'class1'],
# ['red', 'L', 13.5, 'class2'],
# ['blue', 'XL', 15.3, 'class1']])
# df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)
#
# size_mapping = {
# 'XL': 3,
# 'L': 2,
# 'M': 1}
# df['size'] = df['size'].map(size_mapping)
# print(df)
#

# class_mapping = {label:idx for idx,label in
# enumerate(np.unique(df['classlabel']))}
# print(class_mapping)
#
# df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)
#
# inv_class_mapping = {v: k for k, v in class_mapping.items()}
# df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)
#
# from sklearn.preprocessing import LabelEncoder
# class_le = LabelEncoder()
# y = class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))
#
# X = df[['color', 'size', 'price']].values
# color_le = LabelEncoder()
# X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)
#
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features=[0])
# print(ohe.fit_transform(X).toarray())
#
# print(pd.get_dummies(df[['price', 'color', 'size']]))

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol',
'Malic acid', 'Ash',
'Alcalinity of ash', 'Magnesium',
'Total phenols', 'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity', 'Hue',
'OD280/OD315 of diluted wines',
'Proline']
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)