# To Detect wether the type of Breast Cancer with SVM (Support Vector Machine) 

##Project Task

In this study, my task is to classify tumors into malignant (cancerous) or benign (non-cancerous) using features obtained from several cell images.

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

##Attribute Information:

###ID number
Diagnosis (M = malignant, B = benign)
Ten real-valued features are computed for each cell nucleus:

Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter² / area — 1.0)
Concavity (severity of concave portions of the contour)
Concave points (number of concave portions of the contour)
Symmetry
Fractal dimension (“coastline approximation” — 1)

##Importing the necessary libraries

```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
```

##Convert the data in a dataframe
```python
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'],['target_names']))
```

##Visualize the relationship between our features
```python
sns.pairplot(df_cancer, hue = 'target_names', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
```

##Check the correlation between our features
```python
plt.figure(figsize= (40,30))
sns.heatmap(df_cancer.corr(), annot = True)
```

##Taking one of most prominent features 
```python
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target_names', data = df_cancer)
```

##Taking values from Dataframe & Converting to Array
```python
X = df_cancer.drop('target_names', axis=1)
y = df_cancer['target_names']
```

##Splitting our Data into Training & Test sets
```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
```

##Feature Scaling the Data
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)
```

##Fitting Our Training set into the model & Calculating the accuracy of it
```python
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
svc_model = SVC(gamma = 'auto')
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
report_svc = classification_report(y_test,y_pred)
```
This gave an accuracy of 97%

##Fine Tuning our Model in order to give better Classification with the help of GridSearchCV
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,0.001,10,100], 'gamma':[0.1,0.01,0.001,10], 'kernel': ['rbf','sigmoid','poly']}
grid = GridSearchCV(SVC(),param_grid, refit = True, verbose = 4)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)
cm = confusion_matrix(y_test,grid_pred)
report_grid = classification_report(y_test,grid_pred)
```

This gave an accuracy of 98%

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.