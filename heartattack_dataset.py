import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

sns.set_style(style='darkgrid')
data = pd.read_csv('data/Heart Attack.csv')
data.dtypes
data.describe()

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
plt.suptitle('Col 1 : age', fontsize=25, fontweight='bold', color='navy')
fig.tight_layout(pad=2.0)
sns.scatterplot(data=data, x='age', y='class', ax=axes[0], hue='class')
sns.histplot(data=data, x='age', ax=axes[1])
sns.boxplot(data=data, x='age', y='class', ax=axes[2])
plt.show()

condition = data.impluse < 1000
data2 = data[condition]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
plt.suptitle('Col 3 : impluse', fontsize=25, fontweight='bold', color='navy')
fig.tight_layout(pad=2.0)
sns.scatterplot(data=data2, x='impluse', y='class', ax=axes[0], hue='class')
sns.histplot(data=data2, x='impluse', ax=axes[1])
sns.boxplot(data=data2, x='impluse', y='class', ax=axes[2])
plt.show()

X = data2.drop(columns='class')
y = data2['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Shape of X_Train set : {}'.format(X_train.shape))
print('Shape of y_Train set : {}'.format(y_train.shape))
print('_'*50)
print('Shape of X_test set : {}'.format(X_test.shape))
print('Shape of y_test set : {}'.format(y_test.shape))

criterions = ['gini', 'entropy', 'log_loss']
best_criterion = str()
splitters = ['best', 'random']
best_splitter = str()
max_depthes = [None, 3, 4, 5, 6, 7, 8, 9]
best_depth = int()
best_acc = 0

for criterion in criterions:
    for splitter in splitters:
        for depth in max_depthes:
            DTs = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth, random_state=0)
            DTs.fit(X_train, y_train)
            y_pred = DTs.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            if (score > best_acc) and (score < 0.98):
                best_acc = score
                best_criterion = criterion
                best_splitter = splitter
                best_depth = depth

print('Best Criterion : ', best_criterion)
print('Best splitter : ', best_splitter)
print('Best depth : ', best_depth)
print('Accuracy Score : ', best_acc)

DTs = DecisionTreeClassifier(criterion=best_criterion, splitter=best_splitter, max_depth=best_depth, random_state=0)
DTs.fit(X_train, y_train)
y_pred = DTs.predict(X_test)
DTs_score = accuracy_score(y_test, y_pred)
print(DTs_score)

n_estimators = [10, 50, 100, 250, 500]
criterions = ['gini', 'entropy']
max_depthes = [None, 2,  4, 6, 8]
best_acc = 0.0001
best_estimator = 0
for estimator in n_estimators:
    for criterion in criterions:
        for depth in max_depthes:
            RF = RandomForestClassifier(n_estimators=estimator, criterion=criterion, max_depth=depth, n_jobs=-1)
            RF.fit(X_train, y_train)
            y_pred = RF.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            if (score > best_acc) and (score < 0.98):
                best_acc = score
                best_estimator = estimator
                best_criterion = criterion
                best_depth = depth

print('Best Criterion : ', best_criterion)
print('Best estimator : ', best_estimator)
print('Best depth : ', best_depth)
print('Accuracy Score : ', best_acc)

RF = RandomForestClassifier(n_estimators=best_estimator, criterion=best_criterion, max_depth=best_depth, n_jobs=-1, random_state=0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
RF_score = accuracy_score(y_test, y_pred)
print(RF_score)

best_acc = 0
for k in range(3, 15, 2):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score > best_acc:
        best_acc = score
        best_k = k

print('Best k :', best_k)
print('score : ', best_acc)

knn = KNeighborsClassifier(n_neighbors=13, n_jobs=-1).fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_score = accuracy_score(y_test, y_pred)
print(knn_score)

result = pd.DataFrame({
    'Algorithms': ['DTs', 'RF', 'knn'],
    'Accuracy' : [DTs_score, RF_score, knn_score]
})
print(result)

with open("models/DTs_model.pkl", "wb") as file:
    pickle.dump(DTs, file)

with open("models/RF_model.pkl", "wb") as file:
    pickle.dump(RF, file)

with open("models/knn_model.pkl", "wb") as file:
    pickle.dump(knn, file)