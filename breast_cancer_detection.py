import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'],['target_names']))

sns.pairplot(df_cancer, hue = 'target_names', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
plt.figure(figsize= (40,30))
sns.heatmap(df_cancer.corr(), annot = True)

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target_names', data = df_cancer)

X = df_cancer.drop('target_names', axis=1)
y = df_cancer['target_names']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
svc_model = SVC(gamma = 'auto')
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
report_svc = classification_report(y_test,y_pred)


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,0.001,10,100], 'gamma':[0.1,0.01,0.001,10], 'kernel': ['rbf','sigmoid','poly']}
grid = GridSearchCV(SVC(),param_grid, refit = True, verbose = 4)
grid.fit(X_train,y_train)

grid_pred = grid.predict(X_test)
cm = confusion_matrix(y_test,grid_pred)
report_grid = classification_report(y_test,grid_pred)