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

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
print(lr.intercept_)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
'magenta', 'yellow', 'black',
'pink', 'lightgreen', 'lightblue',
'gray', 'indigo', 'orange']
weights, params = [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]),
feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

X_selected = forest.transform(X_train, threshold=0.15)
print(X_selected.shape)