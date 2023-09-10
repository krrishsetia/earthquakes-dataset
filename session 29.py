import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing


pd.options.display.max_columns = 5
pd.options.display.max_rows = 10000000

data = pd.read_csv('Eartquakes-1990-2023.csv')

data.drop(['state','tsunami','place'],axis=1,inplace=True)
def status(var):
    if var == 'reviewed':
        return 0
    elif var == 'automatic':
        return 1

def Type(var):
    if var == 'earthquake':
        return int(0)
    elif var == 'quarry blast':
        return int(1)

data['status'] = data['status'].apply(status)
data['data_type'] = data['data_type'].apply(Type)

data['date sep'] = data['date'].str.split('-')

temp1 = []
temp2 = []

def split(var):
    temp1.append(int(var[0]))
    temp2.append(int(var[1]))
    a = var[2]
    return int(a[0:2])

data['day'] = data['date sep'].apply(split)
data['year'] = pd.Series(temp1)
data['month'] = pd.Series(temp2)

data.drop(['date','date sep'],axis=1,inplace=True)
data.dropna(axis=0,how='any',inplace=True)

x = data['magnitudo'].values.reshape(-1,1)
y = data['data_type'].values.reshape(-1,1)
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()

y_pred_round = np.round(y_pred)
y_test_round = np.round(y_test)
print('mean error squared:',metrics.mean_squared_error(y_true=y_test,y_pred=y_pred))
print('regular accuracy:',metrics.accuracy_score(y_true=y_test_round,y_pred=y_pred_round)*100,'%')
print('balanced accuracy:',metrics.balanced_accuracy_score(y_true=y_test_round,y_pred=y_pred_round)*100,'%')
print('precision:',metrics.precision_score(y_true=y_test_round,y_pred=y_pred_round,average='weighted',zero_division=0)*100,'%')
print('F1:',metrics.f1_score(y_true=y_test_round,y_pred=y_pred_round,average='weighted')*100,'%')
print('kappa:',metrics.cohen_kappa_score(y1=y_test_round,y2=y_pred_round)*100,'%')

matrix = metrics.confusion_matrix(y_true=y_test_round,y_pred=y_pred_round)
display = metrics.ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()